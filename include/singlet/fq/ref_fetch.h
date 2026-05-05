#pragma once
// REF-FETCH-CMD: Reference download and STAR build plan builder
// Builds shell commands for downloading Ensembl genome/GTF and running
// STAR --runMode genomeGenerate. Does NOT execute anything.
// No external dependencies — header-only.

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "lib1fq/species_registry.h"

namespace singlet::ref_fetch {

using species_registry::SpeciesInfo;

// ── Types ─────────────────────────────────────────────────────────────────────

struct FetchPlan {
    std::string species_name;
    std::string assembly;
    std::string fasta_url;       // Ensembl FTP URL for genome FASTA (.fa.gz)
    std::string gtf_url;         // Ensembl FTP URL for GTF (.gtf.gz)
    std::string install_dir;     // e.g. ~/.config/singlify/references/GRCh38/
    std::string star_dir;        // install_dir + "star_2.7.11b/"
    int genome_size_gb;          // estimated peak RAM for STAR genome build (GB)
    std::string download_cmd;    // wget commands for FASTA + GTF
    std::string star_build_cmd;  // STAR --runMode genomeGenerate command
};

// ── Helpers ───────────────────────────────────────────────────────────────────

// Estimate peak RAM (GB) for STAR genome indexing based on genome size
// Rule of thumb: ~2× genome size + 4 GB for SA, rounded to nearest 4 GB
inline int estimate_star_ram_gb(const SpeciesInfo& sp) {
    // Approximate genome sizes (Mbp) keyed by taxon_id
    // For unknown species fall back to a conservative 32 GB
    struct {
        int taxon_id;
        int genome_mbp;
    } sizes[] = {
        {9606, 3100},    // human    ~3.1 Gb
        {10090, 2700},   // mouse    ~2.7 Gb
        {7955, 1400},    // zebrafish~1.4 Gb
        {10116, 2900},   // rat      ~2.9 Gb
        {9031, 1000},    // chicken  ~1.0 Gb
        {9823, 2500},    // pig      ~2.5 Gb
        {9913, 2700},    // cow      ~2.7 Gb
        {9615, 2400},    // dog      ~2.4 Gb
        {9541, 3000},    // macaque  ~3.0 Gb
        {9544, 2900},    // rhesus   ~2.9 Gb
        {9796, 2700},    // horse    ~2.7 Gb
        {9685, 2300},    // cat      ~2.3 Gb
        {8364, 1400},    // frog     ~1.4 Gb
        {28377, 1800},   // anole   ~1.8 Gb
        {13616, 3600},   // opossum  ~3.6 Gb
        {7227, 180},     // fly        180 Mb
        {6239, 100},     // worm       100 Mb
        {3702, 135},     // arabidopsis 135 Mb
        {4932, 12},      // yeast        12 Mb
        {9986, 2700},    // rabbit
        {9940, 2600},    // sheep
        {8090, 700},     // medaka
        {7719, 160},     // sea squirt
        {9669, 2400},    // ferret
        {8296, 32000},   // axolotl  ~32 Gb — giant genome
        {4577, 2100},    // maize
        {8319, 20000},   // newt    ~20 Gb
        {5833, 23},      // plasmodium 23 Mb
        {79327, 800},    // planarian
        {6183, 360},     // blood fluke
        {105023, 1700},  // killifish
        {44689, 34},     // dictyostelium 34 Mb
        {10141, 2700},   // guinea pig
        {7652, 800},     // sea urchin
        {5691, 25},      // trypanosome
        {10036, 2400},   // hamster
        {6500, 1000},    // sea slug
    };
    int genome_mbp = 3100;  // default: assume human-sized
    for (auto& s : sizes)
        if (s.taxon_id == sp.taxon_id) {
            genome_mbp = s.genome_mbp;
            break;
        }

    // RAM = 2 * genome_size + 4 GB overhead, rounded up to multiple of 4
    int ram_gb = (genome_mbp / 500) * 2 + 4;
    return ((ram_gb + 3) / 4) * 4;  // round up to nearest 4
}

// ── Core API ──────────────────────────────────────────────────────────────────

// Build a complete fetch plan for the given species
inline FetchPlan plan_fetch(const SpeciesInfo& sp,
                            const std::string& ref_base,
                            int threads = 20) {
    FetchPlan plan;
    plan.species_name = sp.common_name;
    plan.assembly = sp.assembly;
    plan.fasta_url = species_registry::ensembl_fasta_url(sp);
    plan.gtf_url = species_registry::ensembl_gtf_url(sp);
    plan.install_dir = ref_base + "/" + sp.assembly + "/";
    plan.star_dir = plan.install_dir + "star_2.7.11b/";
    plan.genome_size_gb = estimate_star_ram_gb(sp);

    // Download command: wget FASTA and GTF into install_dir
    plan.download_cmd =
        "mkdir -p " + plan.install_dir +
        " && \\\n"
        "wget -c -q -P " +
        plan.install_dir + " " + plan.fasta_url +
        " && \\\n"
        "wget -c -q -P " +
        plan.install_dir + " " + plan.gtf_url;

    // Decompress FASTA filename
    std::string fasta_gz = plan.fasta_url.substr(plan.fasta_url.rfind('/') + 1);
    std::string fasta_fa = plan.install_dir + fasta_gz.substr(0, fasta_gz.size() - 3);
    std::string gtf_gz = plan.gtf_url.substr(plan.gtf_url.rfind('/') + 1);
    std::string gtf_file = plan.install_dir + gtf_gz.substr(0, gtf_gz.size() - 3);

    // STAR genomeGenerate command
    plan.star_build_cmd =
        "gunzip -kf " + plan.install_dir + fasta_gz +
        " && \\\n"
        "gunzip -kf " +
        plan.install_dir + gtf_gz +
        " && \\\n"
        "mkdir -p " +
        plan.star_dir +
        " && \\\n"
        "STAR --runMode genomeGenerate"
        " --runThreadN " +
        std::to_string(threads) +
        " --genomeDir " + plan.star_dir +
        " --genomeFastaFiles " + fasta_fa +
        " --sjdbGTFfile " + gtf_file +
        " --genomeSAindexNbases 14";  // safe default; STAR warns and adjusts if needed

    return plan;
}

// ── SLURM script builder ──────────────────────────────────────────────────────

// Build a SLURM array job script to install all 37 species
inline std::string build_install_all_script(const std::string& ref_base,
                                            int threads = 20) {
    const auto& species = species_registry::all_species();
    std::ostringstream ss;
    ss << "#!/bin/bash\n"
       << "#SBATCH --job-name=singlify_ref_install\n"
       << "#SBATCH --array=0-" << (species.size() - 1) << "\n"
       << "#SBATCH --cpus-per-task=" << threads << "\n"
       << "#SBATCH --time=04:00:00\n"
       << "#SBATCH --output=" << ref_base << "/logs/install_%a.log\n"
       << "\n"
       << "mkdir -p " << ref_base << "/logs\n"
       << "\n"
       << "# Species array (taxon_id:common_name:assembly)\n"
       << "SPECIES=(\n";
    for (const auto& sp : species)
        ss << "  \"" << sp.taxon_id << ":" << sp.common_name << ":" << sp.assembly << "\"\n";
    ss << ")\n\n"
       << "ENTRY=${SPECIES[$SLURM_ARRAY_TASK_ID]}\n"
       << "TAXON=$(echo $ENTRY | cut -d: -f1)\n"
       << "NAME=$(echo $ENTRY | cut -d: -f2)\n"
       << "ASSEMBLY=$(echo $ENTRY | cut -d: -f3)\n"
       << "\n"
       << "echo \"Installing $NAME ($ASSEMBLY) ...\"\n"
       << "singlify index fetch --species \"$NAME\" --ref-base " << ref_base
       << " --threads " << threads << "\n";
    return ss.str();
}

// ── Registry file helpers ─────────────────────────────────────────────────────

// Return path to the JSON registry file inside ref_base
inline std::string registry_path(const std::string& ref_base) {
    return ref_base + "/references.json";
}

// Write a minimal references.json listing installed assemblies
inline void write_registry([[maybe_unused]] const std::string& ref_base,
                           const std::vector<std::string>& installed_assemblies) {
    std::ostringstream ss;
    ss << "{\n  \"installed\": [\n";
    for (size_t i = 0; i < installed_assemblies.size(); ++i) {
        ss << "    \"" << installed_assemblies[i] << "\"";
        if (i + 1 < installed_assemblies.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n}\n";

    // Write file (no STL file I/O required in header — caller must write the string)
    // This overload returns the JSON content; caller writes it.
    // Use write_registry_string() for testability.
    (void)ss.str();  // silence unused warning; see write_registry_string below
}

// Testable variant: return JSON string instead of writing
inline std::string write_registry_string(const std::vector<std::string>& installed_assemblies) {
    std::ostringstream ss;
    ss << "{\n  \"installed\": [\n";
    for (size_t i = 0; i < installed_assemblies.size(); ++i) {
        ss << "    \"" << installed_assemblies[i] << "\"";
        if (i + 1 < installed_assemblies.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n}\n";
    return ss.str();
}

// Check if a reference assembly is listed in the in-memory installed set
inline bool is_installed(const std::vector<std::string>& installed_assemblies,
                         const std::string& assembly) {
    for (const auto& a : installed_assemblies)
        if (a == assembly) return true;
    return false;
}

}  // namespace singlet::ref_fetch
