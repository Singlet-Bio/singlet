// SPECIES-KMER-DB: Unit tests for species_registry.h
// Run via ctest or standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_sr test/test_species_registry.cpp && /tmp/test_sr

#include <cassert>
#include <iostream>
#include <string>

#include "lib1fq/species_registry.h"

using namespace singlet::species_registry;

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* name) {
        ++passed;
        std::cerr << "PASS: " << name << "\n";
    };
    auto FAIL = [&](const char* name, const char* msg) {
        ++failed;
        std::cerr << "FAIL: " << name << " — " << msg << "\n";
    };

    // ── 1. Total species count ────────────────────────────────────────────────
    {
        auto& sp = all_species();
        if (sp.size() == 37)
            PASS("registry has 37 species");
        else
            FAIL("registry size", ("got " + std::to_string(sp.size())).c_str());
    }

    // ── 2. find_by_taxon ─────────────────────────────────────────────────────
    {
        auto* h = find_by_taxon(9606);
        if (h && std::string(h->common_name) == "human")
            PASS("taxon 9606 = human");
        else
            FAIL("taxon 9606", "not found or wrong name");
    }
    {
        auto* m = find_by_taxon(10090);
        if (m && std::string(m->common_name) == "mouse")
            PASS("taxon 10090 = mouse");
        else
            FAIL("taxon 10090", "not found or wrong name");
    }
    {
        auto* z = find_by_taxon(7955);
        if (z && std::string(z->common_name) == "zebrafish")
            PASS("taxon 7955 = zebrafish");
        else
            FAIL("taxon 7955", "not found or wrong name");
    }
    {
        auto* x = find_by_taxon(99999999);
        if (!x)
            PASS("taxon 99999999 = nullptr");
        else
            FAIL("taxon 99999999", "should be nullptr");
    }

    // ── 3. find_by_name (case-insensitive) ───────────────────────────────────
    {
        auto* h = find_by_name("human");
        if (h && h->taxon_id == 9606)
            PASS("name 'human' → taxon 9606");
        else
            FAIL("name human", "not found");
    }
    {
        auto* h = find_by_name("HUMAN");
        if (h && h->taxon_id == 9606)
            PASS("name 'HUMAN' case-insensitive");
        else
            FAIL("name HUMAN", "case-insensitive match failed");
    }
    {
        auto* m = find_by_name("mouse");
        if (m && m->taxon_id == 10090)
            PASS("name 'mouse' → taxon 10090");
        else
            FAIL("name mouse", "not found");
    }
    {
        auto* f = find_by_name("fruit fly");
        if (f && f->taxon_id == 7227)
            PASS("name 'fruit fly' → taxon 7227");
        else
            FAIL("name fruit fly", "not found");
    }
    {
        auto* x = find_by_name("unicorn");
        if (!x)
            PASS("name 'unicorn' = nullptr");
        else
            FAIL("name unicorn", "should be nullptr");
    }

    // ── 4. find_by_scientific ────────────────────────────────────────────────
    {
        auto* h = find_by_scientific("Homo sapiens");
        if (h && h->taxon_id == 9606)
            PASS("scientific 'Homo sapiens' exact");
        else
            FAIL("scientific Homo sapiens", "not found");
    }
    {
        auto* z = find_by_scientific("Danio");
        if (z && z->taxon_id == 7955)
            PASS("scientific partial 'Danio' = zebrafish");
        else
            FAIL("scientific partial Danio", "not found");
    }
    {
        auto* h = find_by_scientific("homo sapiens");
        if (h && h->taxon_id == 9606)
            PASS("scientific 'homo sapiens' lowercase");
        else
            FAIL("scientific homo sapiens lowercase", "case-insensitive failed");
    }
    {
        auto* x = find_by_scientific("Unicornus fantasticus");
        if (!x)
            PASS("scientific 'Unicornus fantasticus' = nullptr");
        else
            FAIL("scientific unicorn", "should be nullptr");
    }

    // ── 5. Ensembl FTP URL — main Ensembl ────────────────────────────────────
    {
        auto* z = find_by_taxon(7955);  // zebrafish — main Ensembl
        std::string url = ensembl_fasta_url(*z);
        bool has_release = url.find("release-111") != std::string::npos;
        bool has_name = url.find("danio_rerio") != std::string::npos;
        bool is_ftp = url.find("ftp.ensembl.org") != std::string::npos;
        if (has_release && has_name && is_ftp)
            PASS("zebrafish fasta URL has release-111 + danio_rerio");
        else
            FAIL("zebrafish fasta URL", url.c_str());
    }
    {
        auto* z = find_by_taxon(7955);
        std::string url = ensembl_gtf_url(*z);
        bool has_gtf = url.find(".gtf.gz") != std::string::npos;
        bool has_rel = url.find("release-111") != std::string::npos;
        if (has_gtf && has_rel)
            PASS("zebrafish GTF URL is .gtf.gz with release");
        else
            FAIL("zebrafish GTF URL", url.c_str());
    }

    // ── 6. Ensembl FTP URL — EnsemblGenomes plants (arabidopsis) ─────────────
    {
        auto* a = find_by_taxon(3702);  // arabidopsis
        std::string url = ensembl_fasta_url(*a);
        bool has_plants = url.find("ensemblgenomes") != std::string::npos && url.find("plants") != std::string::npos;
        bool has_release = url.find("release-59") != std::string::npos;
        if (has_plants && has_release)
            PASS("arabidopsis fasta URL → EnsemblGenomes plants release-59");
        else
            FAIL("arabidopsis fasta URL", url.c_str());
    }

    // ── 7. Ensembl FTP URL — EnsemblGenomes metazoa (planarian) ──────────────
    {
        auto* p = find_by_taxon(79327);  // planarian
        std::string url = ensembl_fasta_url(*p);
        bool has_metazoa = url.find("metazoa") != std::string::npos;
        if (has_metazoa)
            PASS("planarian fasta URL → EnsemblGenomes metazoa");
        else
            FAIL("planarian fasta URL", url.c_str());
    }

    // ── 8. Ensembl FTP URL — EnsemblGenomes protists (malaria) ───────────────
    {
        auto* mal = find_by_taxon(5833);  // plasmodium
        std::string url = ensembl_fasta_url(*mal);
        bool has_protists = url.find("protists") != std::string::npos;
        if (has_protists)
            PASS("plasmodium fasta URL → EnsemblGenomes protists");
        else
            FAIL("plasmodium fasta URL", url.c_str());
    }

    // ── 9. GENCODE special case URLs (human + mouse) ──────────────────────────
    {
        auto* h = find_by_taxon(9606);
        std::string url = ensembl_fasta_url(*h);
        bool is_grch38 = url.find("GRCh38") != std::string::npos;
        bool is_primary = url.find("primary_assembly") != std::string::npos;
        if (is_grch38 && is_primary)
            PASS("human fasta URL has GRCh38 primary_assembly");
        else
            FAIL("human fasta URL", url.c_str());
    }
    {
        auto* m = find_by_taxon(10090);
        std::string url = ensembl_fasta_url(*m);
        bool is_grcm39 = url.find("GRCm39") != std::string::npos;
        if (is_grcm39)
            PASS("mouse fasta URL has GRCm39");
        else
            FAIL("mouse fasta URL", url.c_str());
    }

    // ── 10. SpeciesInfo field spot-checks ────────────────────────────────────
    {
        auto* rat = find_by_taxon(10116);
        bool ok = rat && std::string(rat->assembly) == "mRatBN7.2" && std::string(rat->ensembl_division) == "ensembl" && rat->ensembl_release == 111;
        if (ok)
            PASS("rat assembly mRatBN7.2, ensembl release 111");
        else
            FAIL("rat fields", "wrong");
    }
    {
        auto* ara = find_by_taxon(3702);
        bool ok = ara && ara->ensembl_release == 59 && std::string(ara->ensembl_division) == "plants";
        if (ok)
            PASS("arabidopsis ensembl_release=59, division=plants");
        else
            FAIL("arabidopsis fields", "wrong");
    }

    // ── Report ────────────────────────────────────────────────────────────────
    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
