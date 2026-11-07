// SPDX-License-Identifier: MIT
// test_mt_caller.cpp
// Unit tests for the reference-based, strand-aware chrM caller (mt_heteroplasmy.h, mt_event_caller.h).
//
//   R1  reference handling: length normalisation (GRCh38 chrM == rCRS 16,569; short / long / lower-case)
//   R2  pool differs from the reference (pool-majority would hide it) -> still called against the reference
//   A1  artefacts rejected: strand-biased alt, uniform noise, single-read noise, masked homopolymer
//   A2  a true clonal heteroplasmy passes every filter, VAF entries are alt/depth
//   H1  homoplasmic donor variant: per-donor haplotype call (AF >= 0.9), not in the public heteroplasmic list
//   H2  per-donor VCF carries DP/AD/AF and lands under private/mt, not mt/
//   S1  mt_strand_consistent
//   C1  cell mask excludes ambient droplets

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/mt_event_caller.h"

namespace fs = std::filesystem;
using namespace singlet;
using namespace singlet::mt;

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg)                                                                              \
    do {                                                                                              \
        if (cond) { ++n_pass; }                                                                       \
        else { ++n_fail; std::cout << "  FAIL: " << msg << " [line " << __LINE__ << "]\n"; }         \
    } while (0)

// ---------------------------------------------------------------------------------------------------------------------
// Synthetic pileup builder: per cell, (pos, base) -> (total, reverse)
// ---------------------------------------------------------------------------------------------------------------------
struct Csc {
    std::vector<int32_t> indptr, indices;
    std::vector<uint16_t> data;
    uint32_t ncols = 0;
    MtCounts view() const { MtCounts v; v.indptr = indptr.data(); v.indices = indices.data(); v.data = data.data(); v.n_cells = ncols; return v; }
};
struct Pile {
    std::vector<std::map<uint32_t, std::pair<uint16_t, uint16_t>>> cells;   // feature -> (total, rev)
    explicit Pile(size_t n) : cells(n) {}
    void add(size_t cell, uint32_t pos, int base, uint16_t fwd, uint16_t rev) {
        auto& e = cells[cell][pos * 4 + base];
        e.first += fwd + rev; e.second += rev;
    }
    Csc total() const { return build(false); }
    Csc reverse() const { return build(true); }
private:
    Csc build(bool rev) const {
        Csc c; c.ncols = static_cast<uint32_t>(cells.size()); c.indptr.assign(c.ncols + 1, 0);
        for (size_t i = 0; i < cells.size(); ++i) {
            for (const auto& kv : cells[i]) {
                const uint16_t v = rev ? kv.second.second : kv.second.first;
                if (v == 0) continue;
                c.indices.push_back(static_cast<int32_t>(kv.first)); c.data.push_back(v);
            }
            c.indptr[i + 1] = static_cast<int32_t>(c.indices.size());
        }
        return c;
    }
};

static std::string make_ref() {
    // deterministic pseudo-random reference over ACGT
    std::string r(MT_LEN, 'A');
    uint32_t x = 12345;
    for (uint32_t i = 0; i < MT_LEN; ++i) { x = x * 1664525u + 1013904223u; r[i] = "ACGT"[(x >> 24) & 3]; }
    return r;
}
static char other_base(char r, int k = 1) {   // k-th base different from r
    int ri = mt_base_index(r);
    return IDX_TO_BASE[(ri + k) & 3];
}
// reference reads of one cell at pos: split roughly 50/50 over strands
static void add_ref(Pile& p, size_t cell, uint32_t pos, const std::string& ref, int n) {
    p.add(cell, pos, mt_base_index(ref[pos]), static_cast<uint16_t>(n - n / 2), static_cast<uint16_t>(n / 2));
}

static const MtSite* find_site(const MtHetResult& r, uint32_t pos0, char alt) {
    for (const auto& s : r.sites) if (s.pos == pos0 && s.alt == alt) return &s;
    return nullptr;
}
static bool in_variants(const MtHetResult& r, uint32_t pos0) {
    for (const auto& v : r.variants) if (v.pos == pos0) return true;
    return false;
}

// ---------------------------------------------------------------------------------------------------------------------
static void test_reference_normalisation() {
    std::cout << "R1 reference normalisation ... ";
    CHECK(normalize_mt_reference("").empty(), "empty stays empty (=> caller falls back / refuses)");
    std::string shortr = "acgtn";
    auto a = normalize_mt_reference(shortr);
    CHECK(a.size() == MT_LEN, "short reference padded to MT_LEN");
    CHECK(a.substr(0, 5) == "ACGTN", "lower-case upper-cased");
    CHECK(a[5] == 'N' && a[MT_LEN - 1] == 'N', "padding is N (never called)");
    auto b = normalize_mt_reference(std::string(MT_LEN + 100, 'C'));
    CHECK(b.size() == MT_LEN, "long reference truncated to MT_LEN");
    auto c = normalize_mt_reference("ARCG");
    CHECK(c[1] == 'N', "IUPAC code -> N");
    std::cout << "done\n";
}

static void test_reference_not_pool_majority() {
    std::cout << "R2 pool differs from reference ... ";
    const std::string ref = make_ref();
    const uint32_t pos = 4000;
    const char alt = other_base(ref[pos]);
    // every cell carries the non-reference base (a 'fixed' difference vs the reference, e.g. rCRS private mutation)
    const size_t n = 60;
    Pile p(n);
    for (size_t c = 0; c < n; ++c) p.add(c, pos, mt_base_index(alt), 6, 6);
    auto tot = p.total(), rv = p.reverse();
    MtCounts t = tot.view(), r = rv.view();

    auto res = call_mt_variants(t, &r, ref);
    CHECK(res.ref_source == "reference", "reference used");
    const MtSite* s = find_site(res, pos, alt);
    CHECK(s != nullptr, "site evaluated against the reference");
    CHECK(s && s->cls == 2, "classified haplotype-like (homoplasmic), not heteroplasmic");
    CHECK(!in_variants(res, pos), "not in the public heteroplasmic list");
    CHECK(res.n_ref_mismatch_pool == 1, "pool majority != reference reported");

    // pool-majority fallback (the old behaviour) cannot see this difference at all
    auto fb = call_mt_variants(t, &r, "");
    CHECK(fb.ref_source == "pool_majority_fallback", "fallback is labelled");
    CHECK(find_site(fb, pos, alt) == nullptr, "pool-majority fallback hides the site (the defect)");
    std::cout << "done\n";
}

static void test_artefacts_and_true_variant() {
    std::cout << "A1/A2 artefacts vs true variant ... ";
    const std::string ref = make_ref();
    const size_t n = 200;
    const uint32_t P_TRUE = 1000, P_STRAND = 2000, P_NOISE1 = 3000, P_UNIFORM = 4000, P_MASK = 309 /*0-based; mask 302-316*/;
    Pile p(n);
    for (size_t c = 0; c < n; ++c) {
        // true clonal heteroplasmy: 40 deep cells carry 12 alt / 28 ref reads (VAF 0.3), alt split over both strands;
        // the other cells are shallow reference cells
        add_ref(p, c, P_TRUE, ref, (c < 40) ? 28 : 12);
        if (c < 40) p.add(c, P_TRUE, mt_base_index(other_base(ref[P_TRUE])), 6, 6);
        // strand artefact: alt only on the forward strand though the reference is on both
        add_ref(p, c, P_STRAND, ref, 12);
        if (c < 40) p.add(c, P_STRAND, mt_base_index(other_base(ref[P_STRAND])), 4, 0);
        // single-read noise in every cell
        add_ref(p, c, P_NOISE1, ref, 11);
        p.add(c, P_NOISE1, mt_base_index(other_base(ref[P_NOISE1])), 1, 0);
        if (c % 2) p.add(c, P_NOISE1, mt_base_index(other_base(ref[P_NOISE1])), 0, 1);   // some on rev
        // uniform low-level noise: 2 alt of 12 in EVERY cell (recurrent error, no clonal structure)
        add_ref(p, c, P_UNIFORM, ref, 10);
        p.add(c, P_UNIFORM, mt_base_index(other_base(ref[P_UNIFORM])), 1, 1);
        // masked homopolymer position with true-looking variant
        add_ref(p, c, P_MASK, ref, 8);
        if (c < 40) p.add(c, P_MASK, mt_base_index(other_base(ref[P_MASK])), 2, 2);
    }
    auto tot = p.total(), rv = p.reverse();
    MtCounts t = tot.view(), r = rv.view();
    auto res = call_mt_variants(t, &r, ref);

    CHECK(in_variants(res, P_TRUE), "true heteroplasmy called");
    const MtSite* st = find_site(res, P_TRUE, other_base(ref[P_TRUE]));
    CHECK(st && st->cls == 1 && st->fail == 0, "true variant passes all filters");
    CHECK(st && st->n_cells_alt == 40, "40 carrier cells");
    CHECK(st && st->n_cells_sig == 40, "all carriers significant against the site background");
    CHECK(st && !std::isnan(st->strand_corr) && st->strand_corr > 0.9f, "per-cell fwd/rev alt VAF correlated");
    CHECK(st && st->overdispersion > 1.5f, "over-dispersed (clonal)");

    const MtSite* ss = find_site(res, P_STRAND, other_base(ref[P_STRAND]));
    CHECK(ss && (ss->fail & kFailStrand) && ss->cls == 0, "forward-only alt rejected by strand filter");
    CHECK(!in_variants(res, P_STRAND), "strand artefact not public");

    const MtSite* sn = find_site(res, P_NOISE1, other_base(ref[P_NOISE1]));
    CHECK(sn == nullptr || sn->cls == 0, "single-read noise not called");
    CHECK(!in_variants(res, P_NOISE1), "single-read noise not public");

    const MtSite* su = find_site(res, P_UNIFORM, other_base(ref[P_UNIFORM]));
    CHECK(su != nullptr && su->cls == 0, "uniform recurrent noise evaluated and rejected");
    CHECK(su && (su->fail & kFailOverdisp), "...because it is not over-dispersed");
    CHECK(!in_variants(res, P_UNIFORM), "uniform noise not public");

    CHECK(find_site(res, P_MASK, other_base(ref[P_MASK])) == nullptr, "masked homopolymer (302-316) never a candidate");
    CHECK(!in_variants(res, P_MASK), "masked site not public");

    // VAF entries: alt / depth for the carrier cells only (carrier depth 40)
    int entries = 0; bool all_ok = true;
    for (uint32_t c = 0; c < res.n_cells; ++c)
        for (int32_t k = res.indptr[c]; k < res.indptr[c + 1]; ++k) {
            ++entries;
            if (std::fabs(res.data[k] - 12.0f / 40.0f) > 1e-5) all_ok = false;
        }
    CHECK(entries == 40, "one VAF entry per carrier cell");
    CHECK(all_ok, "VAF = alt / depth against the reference");
    auto names = het_feature_names(res);
    CHECK(names.size() == 1 && names[0].find(">") != std::string::npos, "feature name is pos:ref>alt");
    std::cout << "done\n";
}

static void test_haplotype_and_vcf() {
    std::cout << "H1/H2 homoplasmic donor variant, VCF fields, private placement ... ";
    const std::string ref = make_ref();
    const size_t per = 50, n = 2 * per;
    const uint32_t P_HAP = 5000, P_HET = 8000, P_LOW = 9000;
    const char hap_alt = other_base(ref[P_HAP]);
    Pile p(n);
    for (size_t c = 0; c < n; ++c) {
        const bool d1 = c >= per;
        // donor 1: homoplasmic alt (AF 1.0); donor 0: reference
        if (d1) p.add(c, P_HAP, mt_base_index(hap_alt), 6, 6); else add_ref(p, c, P_HAP, ref, 12);
        // an inherited-heteroplasmy-like site in donor 0 only (VAF 0.4)
        if (!d1) { add_ref(p, c, P_HET, ref, 24); p.add(c, P_HET, mt_base_index(other_base(ref[P_HET])), 8, 8); }
        else add_ref(p, c, P_HET, ref, 10);
        // donor 1: shallow position (3 reads), below hap_min_depth -> no call
        if (d1 && c < per + 1) p.add(c, P_LOW, mt_base_index(other_base(ref[P_LOW])), 2, 1);
        else add_ref(p, c, P_LOW, ref, 1);
    }
    auto tot = p.total(), rv = p.reverse();
    MtCounts t = tot.view(), r = rv.view();

    std::vector<int> grp(n); for (size_t c = 0; c < n; ++c) grp[c] = c >= per ? 1 : 0;
    auto hap = call_group_haplotypes(t, &r, grp, 2, ref);
    CHECK(hap.size() == 2, "two groups");
    bool d1_has = false, d0_has_hap = false, d0_has_het = false, low_called = false;
    for (const auto& h : hap[1]) { if (h.pos == P_HAP && h.alt == hap_alt) { d1_has = true; CHECK(h.af >= 0.99f && h.dp == per * 12, "AF/DP of the haplotype call"); } if (h.pos == P_LOW) low_called = true; }
    for (const auto& h : hap[0]) { if (h.pos == P_HAP) d0_has_hap = true; if (h.pos == P_HET) d0_has_het = true; }
    CHECK(d1_has, "donor1 homoplasmic variant called (AF >= 0.9)");
    CHECK(!d0_has_hap, "donor0 does not carry it");
    CHECK(!d0_has_het, "heteroplasmic (AF 0.4) site is not a haplotype call");
    CHECK(!low_called, "below hap_min_depth not called");

    // pooled caller: haplotype site is haplotype-like (private), the heteroplasmic one is public
    auto pooled = call_mt_variants(t, &r, ref);
    const MtSite* sh = find_site(pooled, P_HAP, hap_alt);
    CHECK(sh && sh->cls == 2, "pooled caller marks donor-defining site haplotype-like");
    CHECK(!in_variants(pooled, P_HAP), "homoplasmic donor variant is not in public mt_heteroplasmy");
    CHECK(in_variants(pooled, P_HET), "heteroplasmic site is public");

    // full G6 orchestrator: VCF fields + private placement
    fs::path tmp = fs::temp_directory_path() / "singlet_test_mt_caller";
    { std::error_code ec0; fs::remove_all(tmp, ec0); } fs::create_directories(tmp);
    SparseAccumulator<uint16_t>::CSCMatrix mt_csc, mt_rev;
    mt_csc.indptr = tot.indptr; mt_csc.indices = tot.indices; mt_csc.data = tot.data; mt_csc.ncols = tot.ncols; mt_csc.nrows = MT_N_FEATURES;
    mt_rev.indptr = rv.indptr; mt_rev.indices = rv.indices; mt_rev.data = rv.data; mt_rev.ncols = rv.ncols; mt_rev.nrows = MT_N_FEATURES;
    DemuxResult dm; dm.n_donors_k = 2;
    for (size_t c = 0; c < n; ++c) {
        DonorAssignment a; a.donor_id = grp[c]; a.prob_max = 0.99f; a.prob_doublet = 0.f; a.label = "donor" + std::to_string(grp[c]);
        dm.assignments.push_back(a);
    }
    std::vector<std::string> bcs; for (size_t c = 0; c < n; ++c) bcs.push_back("BC" + std::to_string(c));
    write_mt_donor_outputs(tmp.string(), mt_csc, dm, bcs, {}, {}, "", ref, &mt_rev, nullptr, &pooled);
    fs::path vcf1 = tmp / "private" / "mt" / "donor1_mt_variants.vcf";
    CHECK(fs::exists(vcf1), "per-donor VCF written under private/mt");
    CHECK(!fs::exists(tmp / "mt" / "donor1_mt_variants.vcf"), "per-donor VCF NOT in the public mt/ dir");
    CHECK(!fs::exists(tmp / "mt" / "donor1_mt_consensus.fa"), "consensus FASTA NOT in the public mt/ dir");
    CHECK(fs::exists(tmp / "private" / "mt" / "donor1_mt_consensus.fa"), "consensus under private/mt");
    CHECK(fs::exists(tmp / "mt" / "mt_summary.tsv"), "mt_summary.tsv stays public");
    std::string vtxt, v0txt;
    { std::ifstream f(vcf1); std::stringstream ss; ss << f.rdbuf(); vtxt = ss.str(); }
    { std::ifstream f0(tmp / "private" / "mt" / "donor0_mt_variants.vcf"); std::stringstream s0; s0 << f0.rdbuf(); v0txt = s0.str(); }
    std::ostringstream want; want << "chrM\t" << (P_HAP + 1) << "\t.\t" << ref[P_HAP] << "\t" << hap_alt;
    CHECK(vtxt.find(want.str()) != std::string::npos, "VCF line REF is the genome reference, ALT the donor allele");
    CHECK(vtxt.find("DP=600") != std::string::npos && vtxt.find("AF=1") != std::string::npos, "VCF carries DP and AF");
    CHECK(vtxt.find("AD=0,600") != std::string::npos, "VCF carries AD");
    CHECK(v0txt.find("chrM\t") == std::string::npos, "donor0 (reference haplotype) has no variants");
    {
        std::ifstream hf(tmp / "private" / "mt" / "het_site_donor_af.tsv");
        std::string line, row;
        std::ostringstream wantp; wantp << (P_HET + 1) << "\t" << ref[P_HET] << "\t" << other_base(ref[P_HET]) << "\t";
        while (std::getline(hf, line)) if (line.rfind(wantp.str(), 0) == 0) row = line;
        CHECK(!row.empty(), "heteroplasmic site listed in the private per-donor AF table");
        std::vector<std::string> f; std::stringstream rs(row); std::string tok;
        while (std::getline(rs, tok, '\t')) f.push_back(tok);
        CHECK(f.size() >= 10 && f[4] == "0" && f[7] == "1", "...top donor is donor0 and the site is flagged donor-specific");
    }
    std::error_code ec; fs::remove_all(tmp, ec);
    std::cout << "done\n";
}

static void test_reference_private_and_masked_haplotype() {
    std::cout << "H3 reference-private position, strand-skewed reads, masked repeat ... ";
    const std::string ref = make_ref();
    const size_t per = 30, n = 2 * per;
    const uint32_t P_PRIV = 1437, P_REP = 309;   // 0-based; 309 is inside the 302-316 mask
    Pile p(n);
    for (size_t c = 0; c < n; ++c) {
        // every cell of both donors carries the non-reference base, reads only on the forward strand (3' scRNA skew)
        p.add(c, P_PRIV, mt_base_index(other_base(ref[P_PRIV])), 8, 0);
        // donor 1 only: homoplasmic variant in the masked C tract, balanced strands; donor 0 reference
        if (c >= per) p.add(c, P_REP, mt_base_index(other_base(ref[P_REP])), 4, 4);
        else add_ref(p, c, P_REP, ref, 8);
    }
    auto tot = p.total(), rv = p.reverse();
    MtCounts t = tot.view(), r = rv.view();
    std::vector<int> grp(n); for (size_t c = 0; c < n; ++c) grp[c] = c >= per ? 1 : 0;
    auto hap = call_group_haplotypes(t, &r, grp, 2, ref);
    bool priv0 = false, priv1 = false, rep1 = false, rep0 = false;
    for (const auto& h : hap[0]) { if (h.pos == P_PRIV) priv0 = true; if (h.pos == P_REP) rep0 = true; }
    for (const auto& h : hap[1]) { if (h.pos == P_PRIV) priv1 = true; if (h.pos == P_REP) rep1 = true; }
    CHECK(priv0 && priv1, "reference-private position called in every donor despite one-strand reads");
    CHECK(rep1 && !rep0, "homoplasmic donor variant inside the repeat mask is still a haplotype call");
    auto pooled = call_mt_variants(t, &r, ref);
    CHECK(!in_variants(pooled, P_REP), "the masked repeat position is never a heteroplasmic call");
    std::cout << "done\n";
}

static void test_strand_consistency() {
    std::cout << "S1 strand consistency ... ";
    MtCallerConfig cfg;
    CHECK(mt_strand_consistent(10, 10, 100, 100, cfg), "both strands, balanced ref");
    CHECK(!mt_strand_consistent(20, 0, 100, 100, cfg), "alt only fwd where ref is on both");
    CHECK(mt_strand_consistent(0, 20, 5, 200, cfg), "skewed (rev) position, alt follows the reference strand");
    CHECK(!mt_strand_consistent(20, 0, 5, 200, cfg), "skewed (rev) position, alt on the other strand");
    CHECK(!mt_strand_consistent(0, 0, 100, 100, cfg), "no alt reads");
    std::cout << "done\n";
}

static void test_cell_mask() {
    std::cout << "C1 cell mask ... ";
    const std::string ref = make_ref();
    const uint32_t P = 1500; const char alt = other_base(ref[P]);
    const size_t n = 100;
    Pile p(n);
    for (size_t c = 0; c < n; ++c) {
        add_ref(p, c, P, ref, c >= 60 ? 30 : 10);
        if (c >= 60) p.add(c, P, mt_base_index(alt), 5, 5);   // carriers are only in the last 40 cells (ambient droplets)
    }
    auto tot = p.total(), rv = p.reverse();
    MtCounts t = tot.view(), r = rv.view();
    std::vector<uint8_t> mask(n, 0); for (size_t c = 0; c < 60; ++c) mask[c] = 1;
    auto masked = call_mt_variants(t, &r, ref, MtCallerConfig(), &mask);
    CHECK(!in_variants(masked, P), "carriers outside the cell mask contribute no evidence");
    CHECK(masked.indptr.size() == n + 1, "all columns kept");
    auto all = call_mt_variants(t, &r, ref);
    CHECK(in_variants(all, P), "without mask the site is called");
    std::cout << "done\n";
}

int main() {
    std::cout << "=== test_mt_caller ===\n";
    test_reference_normalisation();
    test_reference_not_pool_majority();
    test_artefacts_and_true_variant();
    test_haplotype_and_vcf();
    test_reference_private_and_masked_haplotype();
    test_strand_consistency();
    test_cell_mask();
    std::cout << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
