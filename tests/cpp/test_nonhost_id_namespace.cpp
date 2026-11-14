// SPDX-License-Identifier: MIT
// test_nonhost_id_namespace.cpp - the bacterial and the fungal index each number their species from 1.
//
// Before v1.3.9 the engine joined the two without an offset: every bacterium shared its id with a fungal record, every bacterial record
// was labeled FUNGAL, and the minimizer hits of a fungal rRNA or ITS record with a low id were counted under the bacterium with that id.
// These tests build small indexes whose ids overlap on purpose, write them to files (so the memory-mapped load path runs, as in the
// engine), join them as the engine does (join_fungal_index), and follow reads through the screen, key tagging, kingdom map, EM and names.
//
//   1. a read of the bacterium is BACTERIAL with the bacterium's name and its own id
//   2. a read of a fungal record with a low id is that fungus, FUNGAL, with the fungus's name and its own id
//   3. the two do not pool their counts in the EM
//   4. a viral id shared with both stays a third species and the viral path is unchanged
//   5. a names-only load (the GPU host) names the species the same way
//   6. a fungal index without a bacterial one keeps its ids and the FUNGAL label
//   7. the plain union (no id bits) is unchanged

#include "singlet/pileup/nonhost/nonhost_em.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace singlet::nonhost;

static int g_failed = 0, g_passed = 0;
#define CHECK(name, cond, msg)                                                          \
    do {                                                                                \
        if (cond) { ++g_passed; std::cerr << "PASS: " << (name) << "\n"; }              \
        else { ++g_failed; std::cerr << "FAIL: " << (name) << " - " << (msg) << "\n"; } \
    } while (0)

static std::string rand_seq(std::mt19937& g, int n) {
    static const char b[] = "ACGT";
    std::string s;
    for (int i = 0; i < n; ++i) s += b[g() & 3];
    return s;
}

struct Fixture {
    std::string bact_g[3], fung_g[3], viral_g[3];   // genomes of ids 1..3 in each index
    std::string bact_path, fung_path, viral_path;
};

static std::string tmp(const char* tag) { return std::string("test_nonhost_id_namespace_") + tag + ".snhskidx"; }

static Fixture make_fixture() {
    std::mt19937 g(12345);
    Fixture f;
    const char* bn[3] = {"Escherichia coli", "Proteus mirabilis", "Klebsiella pneumoniae"};
    const char* fn[3] = {"Candida albicans rRNA", "Aspergillus ITS", "Fusarium ITS"};
    const char* vn[3] = {"Human herpesvirus 4", "Influenza A virus", "SARS-CoV-2"};
    auto build = [&](std::string* gen, const char** names, const std::string& path) {
        std::vector<std::pair<std::string, uint32_t>> seqs;
        for (int i = 0; i < 3; ++i) { gen[i] = rand_seq(g, 3000); seqs.push_back({gen[i], static_cast<uint32_t>(i + 1)}); }
        MinSketchIndex idx = MinSketchIndex::build_from_sequences(seqs, 21, 11);
        for (int i = 0; i < 3; ++i) idx.set_species_name(static_cast<uint32_t>(i + 1), names[i]);
        idx.save(path);
    };
    f.bact_path = tmp("bact"); f.fung_path = tmp("fung"); f.viral_path = tmp("viral");
    build(f.bact_g, bn, f.bact_path);
    build(f.fung_g, fn, f.fung_path);
    build(f.viral_g, vn, f.viral_path);
    return f;
}

static std::vector<std::string> reads_of(const std::string& genome, int n, std::mt19937& g) {
    std::vector<std::string> r;
    for (int i = 0; i < n; ++i) r.push_back(genome.substr(g() % (genome.size() - 150), 150));
    return r;
}

struct Joined {
    MinSketchIndex viral, microbial;
    std::unordered_set<uint32_t> fungal;
};

static Joined join(const Fixture& f, bool names_only, bool with_bact = true) {
    auto load = [&](const std::string& p) { return names_only ? MinSketchIndex::load_names(p) : MinSketchIndex::load(p); };
    Joined j{load(f.viral_path), MinSketchIndex(21, 11), {}};
    j.microbial.finalize();
    bool have = false;
    if (with_bact) { j.microbial = load(f.bact_path); have = true; }
    join_fungal_index(j.microbial, have, load(f.fung_path), j.fungal);
    return j;
}

static std::string name_of(const Joined& j, uint32_t key) {
    return is_microbial_key(key) ? j.microbial.species_name(microbial_id(key)) : j.viral.species_name(key);
}

int main() {
    const Fixture f = make_fixture();
    std::mt19937 g(7);
    const Joined j = join(f, false);
    NonHostScreener screener(j.viral, j.microbial, 0.05f, j.fungal);

    // 1 + 2: one read of the bacterium Proteus (id 2 in the bacterial index) and one of the fungal record with id 2
    {
        const std::string bread = reads_of(f.bact_g[1], 1, g)[0], fread = reads_of(f.fung_g[1], 1, g)[0];
        auto bm = screener.classify_multi_batch({bread}, 0.25f, 1);
        auto fm = screener.classify_multi_batch({fread}, 0.25f, 1);
        CHECK("bacterial read has hits", bm.size() == 1 && !bm[0].microbial_hits.empty(), "no microbial hit");
        CHECK("fungal read has hits", fm.size() == 1 && !fm[0].microbial_hits.empty(), "no microbial hit");
        // the ids before tagging: bacterial id 2 plain, fungal id 2 with the fungal bit
        CHECK("bacterial hit id plain", bm[0].microbial_hits.size() == 1 && bm[0].microbial_hits[0].first == 2u, "bacterial id changed");
        CHECK("fungal hit id carries the bit", fm[0].microbial_hits.size() == 1 && fm[0].microbial_hits[0].first == (kFungalIdBit | 2u),
              "fungal id not offset");
        std::vector<NonHostScreener::MultiHit> all = {bm[0], fm[0]};
        CHECK("tagging", tag_microbial_keys(all), "tag refused");
        const auto kmap = build_kingdom_map(all, screener.fungal_species());
        const uint32_t bk = all[0].microbial_hits[0].first, fk = all[1].microbial_hits[0].first;
        CHECK("bacterial label", kmap.at(bk) == NonHostCategory::BACTERIAL, "not BACTERIAL");
        CHECK("bacterial name", name_of(j, bk) == "Proteus mirabilis", name_of(j, bk));
        CHECK("bacterial own id", raw_species_id(bk) == 2u, "raw id wrong");
        CHECK("fungal label", kmap.at(fk) == NonHostCategory::FUNGAL, "not FUNGAL");
        CHECK("fungal name", name_of(j, fk) == "Aspergillus ITS", name_of(j, fk));
        CHECK("fungal own id", raw_species_id(fk) == 2u && is_fungal_key(fk) && !is_fungal_key(bk), "raw id wrong");
        CHECK("single-read classify labels",
              screener.classify(fread).category == NonHostCategory::FUNGAL && screener.classify(bread).category == NonHostCategory::BACTERIAL,
              "classify label");
    }

    // 3 + 4: 60 bacterial, 40 fungal and 100 viral reads, all of id 2: three species, no pooling
    {
        std::vector<std::string> reads;
        for (auto& r : reads_of(f.bact_g[1], 60, g)) reads.push_back(r);
        for (auto& r : reads_of(f.fung_g[1], 40, g)) reads.push_back(r);
        for (auto& r : reads_of(f.viral_g[1], 100, g)) reads.push_back(r);
        auto mh = screener.classify_multi_batch(reads, 0.25f, 2);
        CHECK("viral hit id unchanged", mh[100].viral_hits.size() == 1 && mh[100].viral_hits[0].first == 2u, "viral id changed");
        CHECK("tagging 2", tag_microbial_keys(mh), "tag refused");
        const auto kmap = build_kingdom_map(mh, screener.fungal_species());
        const auto res = em_deconvolve(mh, kmap, 500, 1e-6f, 0.0f, 0.0f, 0.0f);
        CHECK("three species", res.size() == 3, "species count " + std::to_string(res.size()));
        int seen = 0;
        for (const auto& sa : res) {
            const std::string nm = name_of(j, sa.species_id);
            if (nm == "Proteus mirabilis" && sa.kingdom == NonHostCategory::BACTERIAL && sa.n_reads_assigned == 60) ++seen;
            if (nm == "Aspergillus ITS" && sa.kingdom == NonHostCategory::FUNGAL && sa.n_reads_assigned == 40) ++seen;
            if (nm == "Influenza A virus" && sa.kingdom == NonHostCategory::VIRAL && sa.n_reads_assigned == 100) ++seen;
        }
        CHECK("counts not pooled", seen == 3, "per-species counts or labels wrong");
    }

    // 5: the names-only load of the GPU host gives the same ids, labels and names
    {
        const Joined n = join(f, true);
        CHECK("names-only fungal set", n.fungal == j.fungal, "fungal id sets differ");
        CHECK("names-only names",
              n.microbial.species_name(2u) == "Proteus mirabilis" && n.microbial.species_name(kFungalIdBit | 2u) == "Aspergillus ITS" &&
                  n.viral.species_name(2u) == "Influenza A virus",
              "names differ");
    }

    // 6: no bacterial index - fungal ids stay plain, label FUNGAL
    {
        const Joined n = join(f, false, false);
        NonHostScreener s2(n.viral, n.microbial, 0.05f, n.fungal);
        const std::string fread = reads_of(f.fung_g[1], 1, g)[0];
        auto mh = s2.classify_multi_batch({fread}, 0.25f, 1);
        CHECK("fungal-only id plain", mh.size() == 1 && mh[0].microbial_hits.size() == 1 && mh[0].microbial_hits[0].first == 2u, "id offset");
        CHECK("fungal-only label", s2.classify(fread).category == NonHostCategory::FUNGAL, "not FUNGAL");
    }

    // 7: the plain union (no id bits) pools ids as before, and a's name wins
    {
        MinSketchIndex a = MinSketchIndex::load(f.bact_path), b = MinSketchIndex::load(f.fung_path);
        MinSketchIndex u = MinSketchIndex::merge(a, b);
        CHECK("plain union name", u.species_name(2u) == "Proteus mirabilis", "a's name did not win");
        auto hits = u.query(f.fung_g[1].substr(100, 150));
        CHECK("plain union ids", hits.count(2u) == 1 && hits.count(kFungalIdBit | 2u) == 0, "plain union offset ids");
    }

    for (const std::string& p : {f.bact_path, f.fung_path, f.viral_path}) std::remove(p.c_str());
    std::cerr << "\n--- " << g_passed << "/" << (g_passed + g_failed) << " checks passed ---\n";
    return g_failed == 0 ? 0 : 1;
}
