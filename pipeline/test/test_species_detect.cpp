// test/test_species_detect.cpp — Unit tests for species_detect.h
//
// Tests:
//  1. organism_to_genome_tag: taxon_id and organism string mapping
//  2. json_extract_str: basic JSON field extraction
//  3. genome_tag_from_star_dir: parse genomeParameters.txt
//  4. kmer revcomp and canonical functions
//  5. Smoke test: detect() on a real .1fq file (optional, skip if missing)

#include "singlet-pileup/species_detect.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

// ── Test helpers ──────────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", msg); } \
    else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", msg); } \
} while(0)

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_organism_mapping() {
    std::fprintf(stderr, "[test_species_detect] organism mapping...\n");
    using namespace species_detect::detail;

    CHECK(organism_to_genome_tag("", "9606") == "GRCh38",    "taxon 9606 → GRCh38");
    CHECK(organism_to_genome_tag("", "10090") == "GRCm39",   "taxon 10090 → GRCm39");
    CHECK(organism_to_genome_tag("Homo sapiens", "") == "GRCh38",  "Homo sapiens → GRCh38");
    CHECK(organism_to_genome_tag("Mus musculus", "") == "GRCm39",  "Mus musculus → GRCm39");
    CHECK(organism_to_genome_tag("human", "") == "GRCh38",    "human → GRCh38");
    CHECK(organism_to_genome_tag("mouse", "") == "GRCm39",    "mouse → GRCm39");
    CHECK(organism_to_genome_tag("Rattus norvegicus", "10116") == "mRatBN7.2", "rat → mRatBN7.2");
    CHECK(organism_to_genome_tag("chicken", "") == "GRCg7b",  "chicken → GRCg7b");
    CHECK(organism_to_genome_tag("", "7955") == "GRCz11",     "taxon 7955 → GRCz11 (zebrafish)");
    // taxon takes priority
    CHECK(organism_to_genome_tag("unknown", "9606") == "GRCh38", "taxon 9606 overrides organism");
}

static void test_json_extract() {
    std::fprintf(stderr, "[test_species_detect] JSON extraction...\n");
    using namespace species_detect::detail;

    const std::string json = R"({
  "sra_metadata": {
    "organism": "Homo sapiens",
    "taxon_id": "9606",
    "tissue": "blood"
  }
})";
    CHECK(json_extract_str(json, "organism") == "Homo sapiens", "extract organism");
    CHECK(json_extract_str(json, "taxon_id") == "9606",          "extract taxon_id");
    CHECK(json_extract_str(json, "tissue") == "blood",           "extract tissue");
    CHECK(json_extract_str(json, "missing") == "",               "missing key → empty");
}

static void test_kmer_canonical() {
    std::fprintf(stderr, "[test_species_detect] k-mer canonical...\n");
    using namespace species_detect::detail;

    // A 21-mer and its revcomp should canonicalize to the same value
    // Encode: AAAAAAAAAAAAAAAAAAAAAA (21 A's = 0)
    uint64_t all_a = 0ULL;
    uint64_t rc_a  = revcomp_kmer(all_a);  // all T's
    CHECK(canonical(all_a) == canonical(rc_a), "canonical(AAAA) == canonical(TTTT)");

    // Asymmetric k-mer: should return min(kmer, revcomp)
    uint64_t km = 0b00000000000000000001ULL; // ...0001 = A...C
    uint64_t rc = revcomp_kmer(km);
    uint64_t c  = canonical(km);
    CHECK(c == std::min(km, rc), "canonical returns min(km, revcomp)");

    // A k-mer and its revcomp must give the same canonical
    CHECK(canonical(km) == canonical(rc), "canon(km) == canon(rev(km))");
}

static void test_genome_tag_from_star_dir() {
    std::fprintf(stderr, "[test_species_detect] genome_tag_from_star_dir...\n");

    // Write a temporary genomeParameters.txt
    const char* tmp_path = "/tmp/test_species_detect_genome_params.txt";
    {
        FILE* f = std::fopen(tmp_path, "w");
        std::fprintf(f, "### STAR   --runMode genomeGenerate\n");
        std::fprintf(f, "genomeFastaFiles\t/ref/GRCh38-2024-A/fasta/genome.fa\n");
        std::fprintf(f, "versionGenome\t2.7.4a\n");
        std::fclose(f);
    }
    // Create fake genome dir
    const char* fake_dir = "/tmp/test_species_detect_fake_genome_dir";
    ::mkdir(fake_dir, 0755);
    char params_path[512];
    snprintf(params_path, sizeof(params_path), "%s/genomeParameters.txt", fake_dir);
    {
        FILE* f = std::fopen(params_path, "w");
        std::fprintf(f, "genomeFastaFiles\t/ref/GRCh38-2024-A/fasta/genome.fa\n");
        std::fclose(f);
    }
    CHECK(species_detect::genome_tag_from_star_dir(fake_dir) == "GRCh38",
          "genome_tag_from_star_dir detects GRCh38");

    // Test GRCm39
    {
        FILE* f = std::fopen(params_path, "w");
        std::fprintf(f, "genomeFastaFiles\t/ref/GRCm39-2024-A/fasta/genome.fa\n");
        std::fclose(f);
    }
    CHECK(species_detect::genome_tag_from_star_dir(fake_dir) == "GRCm39",
          "genome_tag_from_star_dir detects GRCm39");

    // Missing file → empty
    CHECK(species_detect::genome_tag_from_star_dir("/nonexistent/path") == "",
          "genome_tag_from_star_dir missing dir → empty");

    std::remove(params_path);
    ::rmdir(fake_dir);
    std::remove(tmp_path);
}

static void test_metadata_fallback() {
    std::fprintf(stderr, "[test_species_detect] metadata organism fallback (AUTOFIX-SPECIES-DETECT-ZERO-HITS)...\n");

    // Write a temporary metadata JSON with "Homo sapiens"
    const char* json_path = "/tmp/test_species_detect_meta.json";
    {
        FILE* f = std::fopen(json_path, "w");
        std::fprintf(f, "{\n  \"gsm_id\": \"GSM9999999\",\n  \"organism\": \"Homo sapiens\",\n  \"taxon_id\": 9606\n}\n");
        std::fclose(f);
    }

    // resolve_organism_from_json must resolve Homo sapiens → GRCh38
    {
        std::string gtag, species, taxon;
        bool ok = species_detect::detail::resolve_organism_from_json(json_path, gtag, species, taxon);
        CHECK(ok,                         "resolve_organism_from_json: Homo sapiens → non-empty");
        CHECK(gtag == "GRCh38",           "resolve_organism_from_json: genome_tag=GRCh38");
        CHECK(!species.empty(),           "resolve_organism_from_json: species non-empty");
        CHECK(taxon == "9606",            "resolve_organism_from_json: taxon_id=9606");
    }

    // resolve_organism_from_json must also work for a species only in the registry
    // (Macaca fascicularis uses crab-eating macaque assembly, NOT in local map)
    const char* json_macaque = "/tmp/test_species_detect_macaque.json";
    {
        FILE* f = std::fopen(json_macaque, "w");
        std::fprintf(f, "{\"organism\": \"Macaca fascicularis\", \"taxon_id\": 9541}\n");
        std::fclose(f);
    }
    {
        std::string gtag, species, taxon;
        bool ok = species_detect::detail::resolve_organism_from_json(json_macaque, gtag, species, taxon);
        CHECK(ok,              "resolve: Macaca fascicularis found in registry");
        CHECK(!gtag.empty(),   "resolve: Macaca fascicularis genome_tag non-empty");
        CHECK(taxon == "9541", "resolve: Macaca fascicularis taxon_id=9541");
    }

    // Missing file → false
    {
        std::string gtag, species, taxon;
        bool ok = species_detect::detail::resolve_organism_from_json("/tmp/nonexistent_XXXXX.json",
                                                                       gtag, species, taxon);
        CHECK(!ok, "resolve_organism_from_json: missing file → false");
    }

    // JSON with no organism → false
    const char* json_empty = "/tmp/test_species_detect_empty.json";
    {
        FILE* f = std::fopen(json_empty, "w");
        std::fprintf(f, "{\"gsm_id\": \"GSM0\"}\n");
        std::fclose(f);
        std::string gtag, species, taxon;
        bool ok = species_detect::detail::resolve_organism_from_json(json_empty, gtag, species, taxon);
        CHECK(!ok, "resolve_organism_from_json: no organism → false");
        std::remove(json_empty);
    }

    std::remove(json_path);
    std::remove(json_macaque);
}

static void test_real_file_smoke(int argc, char* argv[]) {
    // Optional: run a real detection on SRR32855204.1fq if path provided
    const char* fq_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fq") == 0 && i + 1 < argc) {
            fq_path = argv[i + 1];
        }
    }
    if (!fq_path) {
        std::fprintf(stderr, "[test_species_detect] (skip real file test — pass --fq <path>)\n");
        return;
    }
    std::fprintf(stderr, "[test_species_detect] real file detect: %s\n", fq_path);
    auto res = species_detect::detect(fq_path, /*verbose=*/true);
    CHECK(res.genome_tag == "GRCh38", "real file → GRCh38");
    CHECK(!res.species.empty(), "real file has species");
    CHECK(res.confident(), "real file is confident");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::fprintf(stderr, "=== species_detect tests ===\n");
    test_organism_mapping();
    test_json_extract();
    test_kmer_canonical();
    test_genome_tag_from_star_dir();
    test_metadata_fallback();
    test_real_file_smoke(argc, argv);

    std::fprintf(stderr, "\n[species_detect] %d/%d passed\n", g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
