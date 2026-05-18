// SPDX-License-Identifier: MIT
// test_mt_events.cpp
// Unit tests for mt_event_caller.h, mt_event_matrix.h, mt_summary.h
// (Gate G6 — donor-aware mitochondrial outputs)
//
// Tests:
//   T1  build_donor_consensus: synthetic 2-donor pileup → expected consensus
//   T2  classify_mt_events: simple substitution → expected event code
//   T3  classify_mt_events: multiple alts → complex event code 9
//   T4  classify_mt_events: alt below heteroplasmy threshold → no event
//   T5  classify_mt_events: alt below min_coverage → no event
//   T6  write_mt_event_matrix: round-trip TP1Z magic check on temp file
//   T7  write_mt_summary: column headers present in output
//   T8  unassigned cell → donor_label "unassigned" in summary
//   T9  pz_meta has all required embedded metadata keys
//   T10 classify_mt_events: code 5 insertion from MtIndelEvent (is_ins=true)
//   T11 classify_mt_events: code 6 deletion from MtIndelEvent (is_ins=false)
//   T12 classify_mt_events: code 7 frameshift (indel in CDS, len%3 != 0)
//   T13 classify_mt_events: code 8 premature stop (sub → stop codon in CDS)
//   T14 classify_mt_events: code 9 from indel+sub at same locus

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "singlet/pileup/mt_event_caller.h"
#include "singlet/pileup/mt_event_matrix.h"
#include "singlet/pileup/mt_summary.h"
#include "singlet/pileup/pz_reader.h"

namespace fs = std::filesystem;
using namespace singlet;
using namespace singlet::mt;

// ---------------------------------------------------------------------------
// Helpers: build a minimal mt CSC for testing
// ---------------------------------------------------------------------------

// Build a CSC matrix representing mt base counts.
// cells = vector of (position, base_idx, count) triples per cell.
struct FakeCell {
    uint32_t pos;
    uint32_t base_idx;   // A=0,C=1,G=2,T=3
    uint16_t count;
};

struct FakeMtCSC {
    std::vector<int32_t>  indptr;
    std::vector<int32_t>  indices;
    std::vector<uint16_t> data;
    uint32_t ncols;
};

FakeMtCSC make_fake_csc(const std::vector<std::vector<FakeCell>>& cells) {
    FakeMtCSC csc;
    csc.ncols = static_cast<uint32_t>(cells.size());
    csc.indptr.resize(csc.ncols + 1, 0);
    for (uint32_t c = 0; c < csc.ncols; ++c) {
        for (const auto& e : cells[c]) {
            csc.indices.push_back(static_cast<int32_t>(e.pos * 4 + e.base_idx));
            csc.data.push_back(e.count);
        }
        csc.indptr[c + 1] = static_cast<int32_t>(csc.indices.size());
    }
    return csc;
}

// Minimal donor assignment builder
DonorAssignment make_assignment(int d) {
    DonorAssignment a;
    a.donor_id   = d;
    a.prob_max   = 0.99f;
    a.prob_doublet = 0.01f;
    if (d >= 0)
        a.label = "donor" + std::to_string(d);
    else
        a.label = "unassigned";
    return a;
}

// ---------------------------------------------------------------------------
// T1: build_donor_consensus
// ---------------------------------------------------------------------------
static void test_build_donor_consensus() {
    std::cout << "T1 build_donor_consensus ... ";

    // 3 cells: cells 0,1 → donor 0 both have 5× 'A' (base 0) at pos 100
    //          cell  2  → donor 1 has 5× 'C' (base 1) at pos 100
    auto csc = make_fake_csc({
        { {100, 0, 5} },          // cell 0: 5×A at pos 100
        { {100, 0, 7} },          // cell 1: 7×A at pos 100
        { {100, 1, 6} },          // cell 2: 6×C at pos 100
    });

    std::vector<DonorAssignment> asgn = {
        make_assignment(0),
        make_assignment(0),
        make_assignment(1),
    };

    auto consensus = build_donor_consensus(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, 2);

    assert(consensus.size() == 2);
    assert(consensus[0][100] == 'A');   // donor0: majority is A (12 counts)
    assert(consensus[1][100] == 'C');   // donor1: majority is C (6 counts)
    // All other positions should be 'N' (no coverage)
    assert(consensus[0][0]   == 'N');
    assert(consensus[1][0]   == 'N');

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T2: simple substitution event code
// ---------------------------------------------------------------------------
static void test_simple_substitution() {
    std::cout << "T2 classify_mt_events substitution ... ";

    // Consensus for donor 0: pos 200 = 'A'
    // Cell 0 (donor 0): pos 200 has 5×A + 3×C → VAF_C = 3/8 = 0.375 ≥ 0.02
    //                   → event code 2 (sub_C)
    auto csc = make_fake_csc({
        { {200, 0, 5}, {200, 1, 3} },   // cell 0
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][200] = 'A';  // donor 0 consensus at pos 200 is A
    std::vector<std::string> barcodes = {"AAACCCTT"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    assert(res.n_cells == 1);
    assert(res.n_loci  == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvSubC);
    assert(res.summaries[0].n_substitutions == 1);
    assert(res.summaries[0].n_loci_with_event == 1);
    assert(res.summaries[0].mt_reads == 8);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T3: multiple alts → complex code 9
// ---------------------------------------------------------------------------
static void test_complex_event() {
    std::cout << "T3 classify_mt_events complex ... ";

    // Consensus: pos 300 = 'A'
    // Cell has 4×A + 3×C + 3×G → two alts both above 2% threshold
    auto csc = make_fake_csc({
        { {300, 0, 4}, {300, 1, 3}, {300, 2, 3} },
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][300] = 'A';
    std::vector<std::string> barcodes = {"CELL0001"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    assert(res.n_loci  == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvComplex);
    // complex events do NOT increment n_substitutions
    assert(res.summaries[0].n_substitutions == 0);
    assert(res.summaries[0].n_loci_with_event == 1);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T4: alt below heteroplasmy threshold → no event
// ---------------------------------------------------------------------------
static void test_below_threshold() {
    std::cout << "T4 classify_mt_events below heteroplasmy threshold ... ";

    // Consensus: pos 400 = 'A'
    // Cell: 100×A + 1×C → VAF_C = 1/101 ≈ 0.0099 < 0.02
    auto csc = make_fake_csc({
        { {400, 0, 100}, {400, 1, 1} },
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][400] = 'A';
    std::vector<std::string> barcodes = {"CELL0002"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    assert(res.n_loci == 0);
    assert(res.data.empty());
    assert(res.summaries[0].n_loci_with_event == 0);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T5: below min_coverage → no event
// ---------------------------------------------------------------------------
static void test_below_min_coverage() {
    std::cout << "T5 classify_mt_events below min_coverage ... ";

    // kMinCoveragePerLocus = 3; total depth here = 2
    auto csc = make_fake_csc({
        { {500, 0, 1}, {500, 1, 1} },   // total = 2 < 3
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][500] = 'A';
    std::vector<std::string> barcodes = {"CELL0003"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    assert(res.n_loci == 0);
    assert(res.data.empty());

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T6: round-trip TP1Z magic check
// ---------------------------------------------------------------------------
static void test_pz_round_trip() {
    std::cout << "T6 write_mt_event_matrix round-trip TP1Z magic ... ";

    // Use the same synthetic data as T2
    auto csc = make_fake_csc({
        { {200, 0, 5}, {200, 1, 3} },
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][200] = 'A';
    std::vector<std::string> barcodes = {"AAACCCTT"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    // Write to temp file
    fs::path tmp = fs::temp_directory_path() / "test_mt_events.1pz";
    bool ok = write_mt_event_matrix(tmp.string(), res, barcodes);
    (void)ok;
    assert(ok);
    assert(fs::exists(tmp));
    assert(fs::file_size(tmp) > 0);

    // Read back and check TP1Z magic
    std::ifstream f(tmp, std::ios::binary);
    assert(f.is_open());
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    // TP1Z in little-endian = 0x5A315054
    assert(magic == 0x5A315054u);

    // Use pz_reader to verify dimensions
    auto rr = singlet::pz::read_1pz(tmp.string());
    assert(rr.m == res.n_loci);
    assert(rr.n == res.n_cells);

    fs::remove(tmp);
    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T7: write_mt_summary column headers
// ---------------------------------------------------------------------------
static void test_mt_summary_headers() {
    std::cout << "T7 write_mt_summary column headers ... ";

    auto csc = make_fake_csc({ { {200, 0, 5}, {200, 1, 3} } });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][200] = 'A';
    std::vector<std::string> barcodes = {"CELL0099"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    fs::path tmp = fs::temp_directory_path() / "test_mt_summary.tsv";
    bool ok = write_mt_summary(tmp.string(), res, barcodes);
    (void)ok;
    assert(ok);
    assert(fs::exists(tmp));

    std::ifstream f(tmp);
    std::string header;
    std::getline(f, header);

    // Check all required columns from DROPLET_OUTPUT_SCHEMA.md §5.4
    const std::vector<std::string> required_cols = {
        "barcode", "donor_id", "mt_reads", "mt_coverage_mean",
        "n_loci_with_event", "n_substitutions", "n_indels",
        "n_frameshifts", "n_premature_stops", "pct_heteroplasmic",
    };
    for (const auto& col : required_cols) {
        (void)col;
        assert(header.find(col) != std::string::npos);
    }

    // Check data row
    std::string row;
    std::getline(f, row);
    assert(row.find("CELL0099") != std::string::npos);
    assert(row.find("donor0")   != std::string::npos);

    fs::remove(tmp);
    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T8: unassigned donor → donor_label "unassigned" in summary
// ---------------------------------------------------------------------------
static void test_unassigned_donor_label() {
    std::cout << "T8 unassigned cell → donor_label in summary ... ";

    auto csc = make_fake_csc({ { {200, 0, 5}, {200, 1, 3} } });
    // Cell is unassigned (donor_id = -1)
    std::vector<DonorAssignment> asgn = { make_assignment(-1) };
    std::vector<std::string> cons = {};  // no donors
    std::vector<std::string> barcodes = {"UNASSIGNED_CELL"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A");

    // No events (no donor consensus to compare against)
    assert(res.n_loci == 0);
    assert(res.summaries[0].donor_label == "unassigned");
    assert(res.summaries[0].n_loci_with_event == 0);
    assert(res.summaries[0].mt_reads == 8);  // reads still counted

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T9: pz_meta has all required embedded metadata keys
// ---------------------------------------------------------------------------
static void test_pz_meta_keys() {
    std::cout << "T9 pz_meta has all required keys ... ";

    auto csc = make_fake_csc({ { {200, 0, 5}, {200, 1, 3} } });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][200] = 'A';
    std::vector<std::string> barcodes = {"CELL_META"};

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "TestBuild-1.0");

    const std::vector<std::string> required_keys = {
        "schema_version", "matrix_kind", "value_dtype", "ref_build",
        "ref_chrom", "donor_consensus", "event_code_map",
        "min_coverage_per_locus", "min_alt_count_per_locus",
        "heteroplasmy_threshold",
    };
    for (const auto& k : required_keys) {
        (void)k;
        assert(res.pz_meta.count(k) == 1);
    }
    assert(res.pz_meta.at("schema_version") == "1.1");
    assert(res.pz_meta.at("matrix_kind")    == "mt_events");
    assert(res.pz_meta.at("value_dtype")    == "uint8");
    assert(res.pz_meta.at("ref_chrom")      == "chrM");
    assert(res.pz_meta.at("ref_build")      == "TestBuild-1.0");
    assert(res.pz_meta.at("heteroplasmy_threshold") == "0.02");
    assert(res.pz_meta.at("min_coverage_per_locus") == "3");
    assert(res.pz_meta.at("min_alt_count_per_locus") == "1");

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T10: code 5 — insertion event
// ---------------------------------------------------------------------------
static void test_insertion_event() {
    std::cout << "T10 classify_mt_events code 5 insertion ... ";

    // Empty CSC (no base counts), cell 0 donor 0
    auto csc = make_fake_csc({ {} });  // cell 0 has zero base counts
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    std::vector<std::string> barcodes = {"INS_CELL"};

    // Inject a 2-base insertion for bc_idx=0 at pos=100
    std::vector<mt::MtIndelEvent> indels = {
        mt::MtIndelEvent{0, 100, 2, true}  // bc_idx=0, pos=100, len=2, is_ins=true
    };

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A",
        indels, {});

    // Should emit exactly one event: code 5 (insertion)
    assert(res.n_loci == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvInsertion);
    assert(res.summaries[0].n_indels == 1);
    assert(res.summaries[0].n_loci_with_event == 1);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T11: code 6 — deletion event
// ---------------------------------------------------------------------------
static void test_deletion_event() {
    std::cout << "T11 classify_mt_events code 6 deletion ... ";

    auto csc = make_fake_csc({ {} });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    std::vector<std::string> barcodes = {"DEL_CELL"};

    // 3-base deletion → len%3==0 → NOT a frameshift, so code 6
    std::vector<mt::MtIndelEvent> indels = {
        mt::MtIndelEvent{0, 200, 3, false}  // bc_idx=0, pos=200, len=3, is_ins=false
    };

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A",
        indels, {});

    assert(res.n_loci == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvDeletion);
    assert(res.summaries[0].n_indels == 1);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T12: code 7 — frameshift (indel inside CDS, len % 3 != 0)
// ---------------------------------------------------------------------------
static void test_frameshift_event() {
    std::cout << "T12 classify_mt_events code 7 frameshift ... ";

    auto csc = make_fake_csc({ {} });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    std::vector<std::string> barcodes = {"FS_CELL"};

    // 1-base insertion → len%3 != 0 → frameshift IF inside a CDS region
    uint32_t test_pos = 300;
    std::vector<mt::MtIndelEvent> indels = {
        mt::MtIndelEvent{0, test_pos, 1, true}  // 1-base insertion = frameshift
    };
    // CDS region covering pos 300
    std::vector<mt::MtCdsRegion> cds = {
        mt::MtCdsRegion{280, 400, 0, false, "ND1"}
    };

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A",
        indels, cds);

    assert(res.n_loci == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvFrameshift);
    assert(res.summaries[0].n_frameshifts == 1);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T13: code 8 — premature stop (substitution introduces stop codon)
// T13 uses the vertebrate mitochondrial code: TAA, TAG, AGA, AGG = STOP
// We set up consensus so pos X = 'A', and cell has alt = 'A' at codon start
// forming TAA (stop) in the positive-strand reading frame.
// Actually easier: pos X is in CDS, consensus = 'G', alt = 'A',
// if codon at (frame=0) becomes TAA with codon[0]=T, [1]=A, [2]=A
// We'd need codon context. For simplicity, test with AGA stop:
//   CDS frame=0, pos 300. Consensus: pos 300='G', pos 301='G', pos 302='A'
//   Cell: pos 300 = 7×G + 3×A (alt A at pos 300)
//   So alt codon = A(300)G(301)A(302) ... wait need to check vertebrate code.
//   AGA = Stop in vertebrate mt code. So:
//   consensus 300='A', 301='G', 302='A'
//   Cell: pos 300 = G alt (codes for AGG at 300? No.)
// Simplest path: use TAA
//   consensus 300='G', 301='A', 302='A'
//   alt at pos 300 = T → codon = T,A,A = TAA = STOP (vertebrate mt and standard)
// ---------------------------------------------------------------------------
static void test_premature_stop_event() {
    std::cout << "T13 classify_mt_events code 8 premature_stop ... ";

    // Build CSC: cell 0, donor 0 with substitution at pos 300
    // 7×G (consensus ref) + 3×T (alt) → VAF_T = 3/10 = 0.3 ≥ 0.02
    auto csc = make_fake_csc({
        { {300, 2, 7}, {300, 3, 3} },   // pos=300: 7×G(=base2) + 3×T(=base3)
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    // Consensus: 300='G', 301='A', 302='A' → codon GAA = Glu (not stop)
    // Alt T at pos 300 → codon TAA = Stop (vertebrate mt code)
    cons[0][300] = 'G';
    cons[0][301] = 'A';
    cons[0][302] = 'A';
    std::vector<std::string> barcodes = {"STOP_CELL"};

    // CDS region covering pos 300-302, frame=0, forward strand
    std::vector<mt::MtCdsRegion> cds = {
        mt::MtCdsRegion{300, 400, 0, false, "ND1"}
    };

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A",
        {}, cds);

    assert(res.n_loci == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvPremStop);
    assert(res.summaries[0].n_premature_stops == 1);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
// T14: code 9 (complex) from combined indel + substitution at same locus
// ---------------------------------------------------------------------------
static void test_complex_indel_sub() {
    std::cout << "T14 classify_mt_events code 9 from indel+sub conflict ... ";

    // Cell 0 has both a substitution AND an insertion at pos 400
    // Substitution: 5×A (ref) + 3×C (alt) → code 2 (sub_C)
    // Insertion: 1-base ins at same pos → code 5
    // Merged: two events at same pos → code 9 (complex)
    auto csc = make_fake_csc({
        { {400, 0, 5}, {400, 1, 3} },   // pos=400: 5×A + 3×C
    });
    std::vector<DonorAssignment> asgn = { make_assignment(0) };
    std::vector<std::string> cons = { std::string(MT_LEN, 'N') };
    cons[0][400] = 'A';
    std::vector<std::string> barcodes = {"COMPLEX2"};

    std::vector<mt::MtIndelEvent> indels = {
        mt::MtIndelEvent{0, 400, 1, true}   // 1-base insertion at same pos
    };

    auto res = classify_mt_events(
        csc.indptr.data(), csc.indices.data(), csc.data.data(),
        csc.ncols, asgn, cons, barcodes, "GRCh38-2024-A",
        indels, {});

    assert(res.n_loci == 1);
    assert(res.data.size() == 1);
    assert(res.data[0] == kEvComplex);
    assert(res.summaries[0].n_loci_with_event == 1);

    std::cout << "PASS\n";
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== test_mt_events ===\n";
    test_build_donor_consensus();
    test_simple_substitution();
    test_complex_event();
    test_below_threshold();
    test_below_min_coverage();
    test_pz_round_trip();
    test_mt_summary_headers();
    test_unassigned_donor_label();
    test_pz_meta_keys();
    test_insertion_event();
    test_deletion_event();
    test_frameshift_event();
    test_premature_stop_event();
    test_complex_indel_sub();
    std::cout << "All tests PASSED.\n";
    return 0;
}
