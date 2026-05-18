// SPDX-License-Identifier: MIT
// REF-FETCH-CMD: Unit tests for ref_fetch.h
// Run via ctest or standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_rf test/test_ref_fetch.cpp && /tmp/test_rf

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/fq/ref_fetch.h"

using namespace singlet::ref_fetch;
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

    const std::string ref_base = "/tmp/singlet_refs";

    // ── 1. Fetch plan — human ─────────────────────────────────────────────────
    const SpeciesInfo* human = find_by_taxon(9606);
    FetchPlan hp = plan_fetch(*human, ref_base, 20);
    {
        if (hp.species_name == "human")
            PASS("human plan species_name == 'human'");
        else
            FAIL("human plan species_name", hp.species_name.c_str());
    }
    {
        if (hp.assembly == "GRCh38")
            PASS("human plan assembly == 'GRCh38'");
        else
            FAIL("human plan assembly", hp.assembly.c_str());
    }
    {
        bool ok = hp.fasta_url.find("GRCh38") != std::string::npos;
        if (ok)
            PASS("human fasta_url contains 'GRCh38'");
        else
            FAIL("human fasta_url", hp.fasta_url.c_str());
    }
    {
        bool ok = hp.gtf_url.find(".gtf.gz") != std::string::npos;
        if (ok)
            PASS("human gtf_url ends in .gtf.gz");
        else
            FAIL("human gtf_url", hp.gtf_url.c_str());
    }
    {
        bool ok = hp.install_dir.find("GRCh38") != std::string::npos && hp.install_dir.find(ref_base) != std::string::npos;
        if (ok)
            PASS("human install_dir contains ref_base + GRCh38");
        else
            FAIL("human install_dir", hp.install_dir.c_str());
    }
    {
        bool ok = hp.star_dir.find("star_2.7.11b") != std::string::npos;
        if (ok)
            PASS("human star_dir contains 'star_2.7.11b'");
        else
            FAIL("human star_dir", hp.star_dir.c_str());
    }
    {
        bool ok = hp.download_cmd.find("wget") != std::string::npos && hp.download_cmd.find("GRCh38") != std::string::npos;
        if (ok)
            PASS("human download_cmd uses wget + GRCh38 URL");
        else
            FAIL("human download_cmd", hp.download_cmd.c_str());
    }
    {
        bool ok = hp.star_build_cmd.find("--runMode genomeGenerate") != std::string::npos && hp.star_build_cmd.find("--runThreadN 20") != std::string::npos && hp.star_build_cmd.find(hp.star_dir) != std::string::npos;
        if (ok)
            PASS("human STAR build cmd has genomeGenerate + 20 threads + star_dir");
        else
            FAIL("human STAR build cmd", hp.star_build_cmd.c_str());
    }
    {
        if (hp.genome_size_gb > 0)
            PASS("human genome_size_gb > 0");
        else
            FAIL("human genome_size_gb", "should be positive");
    }

    // ── 2. Fetch plan — zebrafish ─────────────────────────────────────────────
    const SpeciesInfo* zebra = find_by_taxon(7955);
    FetchPlan zp = plan_fetch(*zebra, ref_base, 16);
    {
        bool ok = zp.fasta_url.find("danio_rerio") != std::string::npos;
        if (ok)
            PASS("zebrafish fasta_url contains 'danio_rerio'");
        else
            FAIL("zebrafish fasta_url", zp.fasta_url.c_str());
    }
    {
        bool ok = zp.star_build_cmd.find("--runThreadN 16") != std::string::npos;
        if (ok)
            PASS("zebrafish STAR cmd uses 16 threads");
        else
            FAIL("zebrafish STAR thread count", zp.star_build_cmd.c_str());
    }

    // ── 3. Fetch plan — arabidopsis (EnsemblGenomes plants) ───────────────────
    const SpeciesInfo* ara = find_by_taxon(3702);
    FetchPlan ap = plan_fetch(*ara, ref_base, 8);
    {
        bool ok = ap.download_cmd.find("ensemblgenomes") != std::string::npos;
        if (ok)
            PASS("arabidopsis download_cmd uses ensemblgenomes URL");
        else
            FAIL("arabidopsis download_cmd", ap.download_cmd.c_str());
    }
    {
        bool ok = ap.fasta_url.find("plants") != std::string::npos;
        if (ok)
            PASS("arabidopsis fasta_url path contains 'plants'");
        else
            FAIL("arabidopsis fasta_url", ap.fasta_url.c_str());
    }

    // ── 4. SLURM script generation ────────────────────────────────────────────
    {
        std::string script = build_install_all_script(ref_base, 20);
        bool has_sbatch = script.find("#SBATCH") != std::string::npos;
        bool has_array = script.find("--array=0-") != std::string::npos;
        bool has_cpus = script.find("cpus-per-task=20") != std::string::npos;
        bool has_singlet = script.find("singlet index fetch") != std::string::npos;
        bool has_human = script.find("9606:human") != std::string::npos;
        if (has_sbatch && has_array && has_cpus && has_singlet && has_human)
            PASS("SLURM script has #SBATCH, array, cpus, singlet index fetch, human entry");
        else
            FAIL("SLURM script", script.substr(0, 200).c_str());
    }
    {
        std::string script = build_install_all_script(ref_base, 20);
        // Array should cover all 37 species (0-36)
        bool ok = script.find("--array=0-36") != std::string::npos;
        if (ok)
            PASS("SLURM array covers all 37 species (0-36)");
        else
            FAIL("SLURM array size", "expected --array=0-36");
    }

    // ── 5. Registry helpers ───────────────────────────────────────────────────
    {
        std::vector<std::string> installed = {"GRCh38", "GRCm39", "GRCz11"};
        std::string json = write_registry_string(installed);
        bool ok = json.find("GRCh38") != std::string::npos && json.find("GRCm39") != std::string::npos && json.find("GRCz11") != std::string::npos;
        if (ok)
            PASS("write_registry_string contains all 3 assemblies");
        else
            FAIL("write_registry_string", json.c_str());
    }
    {
        std::string json = write_registry_string({});
        bool ok = json.find("installed") != std::string::npos;
        if (ok)
            PASS("write_registry_string empty → valid JSON skeleton");
        else
            FAIL("write_registry_string empty", json.c_str());
    }
    {
        std::vector<std::string> installed = {"GRCh38", "GRCm39"};
        if (is_installed(installed, "GRCh38"))
            PASS("is_installed: GRCh38 is installed");
        else
            FAIL("is_installed GRCh38", "should be true");
    }
    {
        std::vector<std::string> installed = {"GRCh38", "GRCm39"};
        if (!is_installed(installed, "GRCz11"))
            PASS("is_installed: GRCz11 not installed");
        else
            FAIL("is_installed GRCz11", "should be false");
    }
    {
        std::vector<std::string> empty;
        if (!is_installed(empty, "GRCh38"))
            PASS("is_installed: empty list → false");
        else
            FAIL("is_installed empty", "should be false");
    }

    // ── Report ────────────────────────────────────────────────────────────────
    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
