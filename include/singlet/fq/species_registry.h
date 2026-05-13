#pragma once
// SPECIES-KMER-DB: Compile-time registry of all 37 species singlet supports
// Derived from scgeo/config/species.py + Ensembl release 111 / EnsemblGenomes release 59
// No external dependencies — header-only.

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace singlet::species_registry {

struct SpeciesInfo {
    int taxon_id;
    const char* common_name;       // "human", "mouse", "zebrafish"
    const char* scientific_name;   // "Homo sapiens", "Mus musculus"
    const char* assembly;          // "GRCh38", "GRCm39"
    const char* annotation;        // "GENCODE v45", "Ensembl 111"
    const char* ensembl_name;      // Ensembl species directory name (lowercase, underscored)
    const char* ensembl_division;  // "ensembl", "plants", "metazoa", "protists"
    int ensembl_release;           // 111 for main Ensembl, 59 for EnsemblGenomes divisions
    bool has_local_index;          // true if reference is installed in SINGLET_REF_BASE
};

// All 37 supported species (from scgeo/config/species.py)
inline const std::vector<SpeciesInfo>& all_species() {
    static const std::vector<SpeciesInfo> species = {
        // ── Original 15 (commonly have local indices) ──
        {9606, "human", "Homo sapiens", "GRCh38", "GENCODE v45", "homo_sapiens", "ensembl", 111, false},
        {10090, "mouse", "Mus musculus", "GRCm39", "GENCODE vM34", "mus_musculus", "ensembl", 111, false},
        {7955, "zebrafish", "Danio rerio", "GRCz11", "Ensembl 111", "danio_rerio", "ensembl", 111, false},
        {10116, "rat", "Rattus norvegicus", "mRatBN7.2", "Ensembl 111", "rattus_norvegicus", "ensembl", 111, false},
        {9031, "chicken", "Gallus gallus", "GRCg7b", "Ensembl 111", "gallus_gallus", "ensembl", 111, false},
        {9823, "pig", "Sus scrofa", "Sscrofa11.1", "Ensembl 111", "sus_scrofa", "ensembl", 111, false},
        {9913, "cow", "Bos taurus", "ARS-UCD1.3", "Ensembl 111", "bos_taurus", "ensembl", 111, false},
        {9615, "dog", "Canis lupus familiaris", "ROS_Cfam_1.0", "Ensembl 111", "canis_lupus_familiaris", "ensembl", 111, false},
        {9541, "crab-eating macaque", "Macaca fascicularis", "Macaca_fascicularis_6.0", "Ensembl 111", "macaca_fascicularis", "ensembl", 111, false},
        {9544, "rhesus macaque", "Macaca mulatta", "Mmul_10", "Ensembl 111", "macaca_mulatta", "ensembl", 111, false},
        {9796, "horse", "Equus caballus", "EquCab3.0", "Ensembl 111", "equus_caballus", "ensembl", 111, false},
        {9685, "cat", "Felis catus", "Felis_catus_9.0", "Ensembl 111", "felis_catus", "ensembl", 111, false},
        {8364, "frog", "Xenopus tropicalis", "UCB_Xtro_10.0", "Ensembl 111", "xenopus_tropicalis", "ensembl", 111, false},
        {28377, "green anole", "Anolis carolinensis", "AnoCar2.0", "Ensembl 111", "anolis_carolinensis", "ensembl", 111, false},
        {13616, "opossum", "Monodelphis domestica", "ASM229v1", "Ensembl 111", "monodelphis_domestica", "ensembl", 111, false},
        // ── Extended species ──
        {7227, "fruit fly", "Drosophila melanogaster", "BDGP6.46", "Ensembl 111", "drosophila_melanogaster", "ensembl", 111, false},
        {6239, "nematode", "Caenorhabditis elegans", "WBcel235", "Ensembl 111", "caenorhabditis_elegans", "ensembl", 111, false},
        {3702, "arabidopsis", "Arabidopsis thaliana", "TAIR10", "Ensembl Plants 59", "arabidopsis_thaliana", "plants", 59, false},
        {4932, "yeast", "Saccharomyces cerevisiae", "R64-1-1", "Ensembl 111", "saccharomyces_cerevisiae", "ensembl", 111, false},
        {9986, "rabbit", "Oryctolagus cuniculus", "OryCun2.0", "Ensembl 111", "oryctolagus_cuniculus", "ensembl", 111, false},
        {9940, "sheep", "Ovis aries", "ARS-UI_Ramb_v2.0", "Ensembl 111", "ovis_aries", "ensembl", 111, false},
        {8090, "medaka", "Oryzias latipes", "ASM223467v1", "Ensembl 111", "oryzias_latipes", "ensembl", 111, false},
        {7719, "sea squirt", "Ciona intestinalis", "KH", "Ensembl 111", "ciona_intestinalis", "ensembl", 111, false},
        {9669, "ferret", "Mustela putorius furo", "MusPutFur1.0", "Ensembl 111", "mustela_putorius_furo", "ensembl", 111, false},
        {8296, "axolotl", "Ambystoma mexicanum", "AmexG_v6.0-DD", "Ensembl 111", "ambystoma_mexicanum", "ensembl", 111, false},
        {4577, "maize", "Zea mays", "Zm-B73-REFERENCE-NAM-5.0", "Ensembl Plants 59", "zea_mays", "plants", 59, false},
        {8319, "iberian ribbed newt", "Pleurodeles waltl", "pwalHiC1", "Ensembl 111", "pleurodeles_waltl", "ensembl", 111, false},
        {5833, "malaria parasite", "Plasmodium falciparum", "ASM276v2", "Ensembl Protists 59", "plasmodium_falciparum", "protists", 59, false},
        {79327, "planarian", "Schmidtea mediterranea", "ASM2634887v1", "Ensembl Metazoa 59", "schmidtea_mediterranea", "metazoa", 59, false},
        {6183, "blood fluke", "Schistosoma mansoni", "Smansoni_v7", "Ensembl Metazoa 59", "schistosoma_mansoni", "metazoa", 59, false},
        {105023, "killifish", "Nothobranchius furzeri", "Nfu_20140520", "Ensembl 111", "nothobranchius_furzeri", "ensembl", 111, false},
        {44689, "social amoeba", "Dictyostelium discoideum", "dicty_2.7", "Ensembl Protists 59", "dictyostelium_discoideum", "protists", 59, false},
        {10141, "guinea pig", "Cavia porcellus", "Cavpor3.0", "Ensembl 111", "cavia_porcellus", "ensembl", 111, false},
        {7652, "sea urchin", "Lytechinus variegatus", "Lvar_3.0", "Ensembl Metazoa 59", "lytechinus_variegatus", "metazoa", 59, false},
        {5691, "trypanosome", "Trypanosoma brucei brucei", "TryBru_Apr2005_chr11", "Ensembl Protists 59", "trypanosoma_brucei", "protists", 59, false},
        {10036, "hamster", "Mesocricetus auratus", "MesAur1.0", "Ensembl 111", "mesocricetus_auratus", "ensembl", 111, false},
        {6500, "sea slug", "Aplysia californica", "AplCal3.0", "Ensembl Metazoa 59", "aplysia_californica", "metazoa", 59, false},
    };
    return species;
}

// ── Lookup helpers ────────────────────────────────────────────────────────────

// Lookup by NCBI taxon ID
inline const SpeciesInfo* find_by_taxon(int taxon_id) {
    for (const auto& sp : all_species())
        if (sp.taxon_id == taxon_id) return &sp;
    return nullptr;
}

// Case-insensitive string comparison helper
inline bool iequal(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    return true;
}

// Lookup by common name (case-insensitive exact match)
inline const SpeciesInfo* find_by_name(std::string_view name) {
    for (const auto& sp : all_species())
        if (iequal(name, sp.common_name)) return &sp;
    return nullptr;
}

// Lookup by scientific name (case-insensitive, full or partial substring match)
inline const SpeciesInfo* find_by_scientific(std::string_view query) {
    // Lowercase query for comparison
    std::string q_lower(query);
    for (auto& c : q_lower) c = (char)std::tolower((unsigned char)c);

    for (const auto& sp : all_species()) {
        std::string sci_lower(sp.scientific_name);
        for (auto& c : sci_lower) c = (char)std::tolower((unsigned char)c);
        if (sci_lower.find(q_lower) != std::string::npos) return &sp;
    }
    return nullptr;
}

// ── URL construction ──────────────────────────────────────────────────────────

// Construct the Ensembl FTP URL for genome FASTA (toplevel, soft-masked)
inline std::string ensembl_fasta_url(const SpeciesInfo& sp) {
    std::string name_cap = sp.ensembl_name;
    if (!name_cap.empty()) name_cap[0] = (char)std::toupper((unsigned char)name_cap[0]);

    // GENCODE special cases for human and mouse
    if (sp.taxon_id == 9606) {
        // Human: GRCh38 primary assembly from GENCODE / Ensembl
        return "https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/dna/"
               "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz";
    }
    if (sp.taxon_id == 10090) {
        // Mouse: GRCm39 primary assembly from GENCODE / Ensembl
        return "https://ftp.ensembl.org/pub/release-111/fasta/mus_musculus/dna/"
               "Mus_musculus.GRCm39.dna.primary_assembly.fa.gz";
    }

    std::string base;
    if (std::string(sp.ensembl_division) == "ensembl") {
        base = "https://ftp.ensembl.org/pub/release-" + std::to_string(sp.ensembl_release) + "/fasta/" + sp.ensembl_name + "/dna/";
    } else {
        base = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/" + std::string(sp.ensembl_division) + "/release-" + std::to_string(sp.ensembl_release) + "/fasta/" + sp.ensembl_name + "/dna/";
    }
    return base + name_cap + "." + sp.assembly + ".dna.toplevel.fa.gz";
}

// Construct the Ensembl FTP URL for GTF annotation
inline std::string ensembl_gtf_url(const SpeciesInfo& sp) {
    std::string name_cap = sp.ensembl_name;
    if (!name_cap.empty()) name_cap[0] = (char)std::toupper((unsigned char)name_cap[0]);

    // GENCODE special cases for human and mouse
    if (sp.taxon_id == 9606) {
        return "https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/"
               "Homo_sapiens.GRCh38.111.gtf.gz";
    }
    if (sp.taxon_id == 10090) {
        return "https://ftp.ensembl.org/pub/release-111/gtf/mus_musculus/"
               "Mus_musculus.GRCm39.111.gtf.gz";
    }

    std::string base;
    if (std::string(sp.ensembl_division) == "ensembl") {
        base = "https://ftp.ensembl.org/pub/release-" + std::to_string(sp.ensembl_release) + "/gtf/" + sp.ensembl_name + "/";
    } else {
        base = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/" + std::string(sp.ensembl_division) + "/release-" + std::to_string(sp.ensembl_release) + "/gtf/" + sp.ensembl_name + "/";
    }
    return base + name_cap + "." + sp.assembly + "." + std::to_string(sp.ensembl_release) + ".gtf.gz";
}

}  // namespace singlet::species_registry
