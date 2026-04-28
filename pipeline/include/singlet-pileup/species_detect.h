// singlet-pileup/species_detect.h — N1: Species auto-detection
//
// Determines the reference species (organism) from a .1fq file so that
// --genome-dir can be resolved automatically.
//
// Three detection paths, tried in order:
//
//   1. Metadata path (zero-cost, <1ms):
//      Read the .1fq metadata JSON → sra_metadata.organism / .taxon_id.
//      Works for all SRA-downloaded files (organism embedded at encode time).
//
//   2. Bloom filter path (~1-2s):
//      Sample R2 reads from the first block, count k-mer hits against
//      species-specific Bloom filters (45MB each) built from the full
//      transcriptome. Bloom filters are loaded from species_filters/ at
//      runtime. Set bloom_filter_dir to use custom locations.
//      Build filters with scripts/build_species_bloom_filters.py.
//
//   3. Diagnostic k-mer path (fallback, <1s, only if Bloom filters missing):
//      Uses a small embedded set of species-specific housekeeping gene
//      k-mers (from species_kmer_db_gen.h). Less accurate on short/ARC reads.
//
// Usage:
//   auto res = species_detect::detect(onefq_path);
//   if (res.confident()) {
//       genome_dir = res.suggested_genome_dir("/path/to/reference/base");
//   }
//
// Bloom filter location: singlify/species_filters/ (relative to binary), or
// set SINGLIFY_SPECIES_FILTERS env var to override.

#pragma once

#include "../lib1fq/reader.h"
#include "../lib1fq/species_registry.h"
#include "bloom_filter.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

// Include generated k-mer DB if it exists (optional — fallback only)
#if __has_include("species_kmer_db_gen.h")
#   include "species_kmer_db_gen.h"
#   define SINGLET_HAVE_KMER_DB 1
#else
#   define SINGLET_HAVE_KMER_DB 0
#endif

namespace species_detect {

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int   KMER_K              = 21;
static constexpr int   SAMPLE_READS        = 5000;   // R2 reads to sample
static constexpr float CONFIDENCE_THRESH   = 0.001f; // min hit-rate to call species
static constexpr float HIGH_CONF_THRESH    = 0.01f;  // "high confidence" threshold
static constexpr float MIN_RATIO           = 2.0f;   // winner must have 2x more hits than runner-up

// ── Result ────────────────────────────────────────────────────────────────────

struct DetectionResult {
    std::string species;       // e.g. "Homo sapiens", "Mus musculus", ""
    std::string taxon_id;      // NCBI taxon, e.g. "9606", "10090"
    std::string genome_tag;    // short tag: "GRCh38", "GRCm39", "unknown"
    float       confidence = 0.0f;
    std::string method;        // "metadata", "kmer", "none"
    int         reads_sampled = 0;
    int         kmers_tested  = 0;
    int         kmers_hit     = 0;

    bool confident() const { return confidence >= CONFIDENCE_THRESH || method == "metadata"; }
    bool high_confidence() const { return confidence >= HIGH_CONF_THRESH || method == "metadata"; }

    // Given a base directory containing reference subdirectories per the
    // Cellarium layout (e.g. /mnt/projects/.../reference/), return the
    // STAR genome subdirectory for the detected species.
    // Returns "" if species unknown.
    std::string suggested_star_dir(const std::string& ref_base) const {
        // Human and mouse have "-2024-A" naming convention
        if (genome_tag == "GRCh38")
            return ref_base + "/GRCh38-2024-A/star_2.7.11b";
        if (genome_tag == "GRCm39")
            return ref_base + "/GRCm39-2024-A/star_2.7.11b";
        // All other species: ref_base/{genome_tag}/star_2.7.11b
        if (!genome_tag.empty() && genome_tag != "unknown") {
            // Try exact tag first
            std::string candidate = ref_base + "/" + genome_tag + "/star_2.7.11b";
            struct stat st;
            if (stat(candidate.c_str(), &st) == 0) return candidate;
            // Try star/ subdirectory
            candidate = ref_base + "/" + genome_tag + "/star";
            if (stat(candidate.c_str(), &st) == 0) return candidate;
        }
        return "";
    }

    void print() const {
        std::fprintf(stderr,
            "[species-detect] species=%s genome=%s confidence=%.3f method=%s",
            species.empty() ? "unknown" : species.c_str(),
            genome_tag.empty() ? "unknown" : genome_tag.c_str(),
            confidence, method.c_str());
        if (method == "kmer")
            std::fprintf(stderr, " (%d hits / %d kmers sampled from %d reads)",
                         kmers_hit, kmers_tested, reads_sampled);
        std::fprintf(stderr, "\n");
    }
};

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace detail {

// Map organism string or taxon_id → genome_tag
inline std::string organism_to_genome_tag(const std::string& organism,
                                           const std::string& taxon_id) {
    // taxon_id takes priority (canonical)
    if (taxon_id == "9606")  return "GRCh38";
    if (taxon_id == "10090") return "GRCm39";
    if (taxon_id == "7955")  return "GRCz11";    // zebrafish
    if (taxon_id == "10116") return "mRatBN7.2";  // rat
    if (taxon_id == "9031")  return "GRCg7b";     // chicken
    if (taxon_id == "9823")  return "Sscrofa11.1"; // pig
    if (taxon_id == "9913")  return "ARS-UCD1.3";  // cow
    if (taxon_id == "9615")  return "ROS_Cfam_1.0"; // dog
    if (taxon_id == "9544")  return "Mmul_10";     // macaque
    if (taxon_id == "7227")  return "BDGP6.46";    // drosophila
    if (taxon_id == "6239")  return "WBcel235";    // C. elegans
    if (taxon_id == "9685")  return "Felis_catus_9.0"; // cat
    if (taxon_id == "9986")  return "OryCun2.0";   // rabbit
    if (taxon_id == "9796")  return "EquCab3.0";   // horse
    if (taxon_id == "8364")  return "UCB_Xtro_10.0"; // frog (X. tropicalis)
    if (taxon_id == "9940")  return "ARS-UI_Ramb_v2.0"; // sheep
    if (taxon_id == "4932" || taxon_id == "559292") return "R64-1-1"; // yeast

    // Fallback: organism string match (case-insensitive)
    auto lo = organism;
    std::transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
    
    // Human / Mouse
    if (lo.find("homo sapiens") != std::string::npos || lo == "human") return "GRCh38";
    if (lo.find("mus musculus") != std::string::npos || lo == "mouse") return "GRCm39";
    
    // Vertebrates
    if (lo.find("danio rerio") != std::string::npos || lo == "zebrafish") return "GRCz11";
    if (lo.find("rattus norvegicus") != std::string::npos || lo == "rat") return "mRatBN7.2";
    if (lo.find("gallus gallus") != std::string::npos || lo == "chicken") return "GRCg7b";
    if (lo.find("sus scrofa") != std::string::npos || lo == "pig") return "Sscrofa11.1";
    if (lo.find("bos taurus") != std::string::npos || lo == "cow" || lo == "cattle") return "ARS-UCD1.3";
    if (lo.find("canis") != std::string::npos || lo == "dog") return "ROS_Cfam_1.0";
    if (lo.find("macaca") != std::string::npos || lo == "macaque") return "Mmul_10";
    if (lo.find("felis catus") != std::string::npos || lo == "cat") return "Felis_catus_9.0";
    if (lo.find("oryctolagus") != std::string::npos || lo == "rabbit") return "OryCun2.0";
    if (lo.find("equus caballus") != std::string::npos || lo == "horse") return "EquCab3.0";
    if (lo.find("xenopus") != std::string::npos || lo == "frog") return "UCB_Xtro_10.0";
    if (lo.find("ovis aries") != std::string::npos || lo == "sheep") return "ARS-UI_Ramb_v2.0";
    
    // Invertebrates
    if (lo.find("drosophila") != std::string::npos || lo == "fly") return "BDGP6.46";
    if (lo.find("caenorhabditis") != std::string::npos || lo == "c_elegans" || lo == "worm") return "WBcel235";
    
    // Yeast
    if (lo.find("saccharomyces") != std::string::npos || lo == "yeast") return "R64-1-1";
    
    return "";
}

// Genome tag → canonical species name
inline std::string genome_tag_to_species(const std::string& tag) {
    if (tag == "GRCh38") return "Homo sapiens";
    if (tag == "GRCm39") return "Mus musculus";
    if (tag == "GRCz11") return "Danio rerio";
    if (tag == "mRatBN7.2") return "Rattus norvegicus";
    if (tag == "GRCg7b") return "Gallus gallus";
    if (tag == "Sscrofa11.1") return "Sus scrofa";
    if (tag == "ARS-UCD1.3") return "Bos taurus";
    if (tag == "ROS_Cfam_1.0") return "Canis lupus familiaris";
    if (tag == "Mmul_10") return "Macaca mulatta";
    if (tag == "BDGP6.46") return "Drosophila melanogaster";
    if (tag == "WBcel235") return "Caenorhabditis elegans";
    if (tag == "Felis_catus_9.0") return "Felis catus";
    if (tag == "OryCun2.0") return "Oryctolagus cuniculus";
    if (tag == "EquCab3.0") return "Equus caballus";
    if (tag == "UCB_Xtro_10.0") return "Xenopus tropicalis";
    if (tag == "ARS-UI_Ramb_v2.0") return "Ovis aries";
    if (tag == "R64-1-1") return "Saccharomyces cerevisiae";
    return "";
}

// Genome tag → taxon_id
inline std::string genome_tag_to_taxon(const std::string& tag) {
    if (tag == "GRCh38") return "9606";
    if (tag == "GRCm39") return "10090";
    if (tag == "GRCz11") return "7955";
    if (tag == "mRatBN7.2") return "10116";
    if (tag == "GRCg7b") return "9031";
    if (tag == "Sscrofa11.1") return "9823";
    if (tag == "ARS-UCD1.3") return "9913";
    if (tag == "ROS_Cfam_1.0") return "9615";
    if (tag == "Mmul_10") return "9544";
    if (tag == "BDGP6.46") return "7227";
    if (tag == "WBcel235") return "6239";
    if (tag == "Felis_catus_9.0") return "9685";
    if (tag == "OryCun2.0") return "9986";
    if (tag == "EquCab3.0") return "9796";
    if (tag == "UCB_Xtro_10.0") return "8364";
    if (tag == "ARS-UI_Ramb_v2.0") return "9940";
    if (tag == "R64-1-1") return "4932";
    return "";
}

// Simple JSON string extractor (no dependency on full JSON parser)
inline std::string json_extract_str(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// Read organism/taxon_id from an external JSON file and resolve to (genome_tag,
// species, taxon_id).  Tries organism_to_genome_tag() first (fast coverage for
// ~20 common species) then falls back to species_registry::find_by_scientific()
// which covers all 37 registered assemblies.  Returns true on success.
inline bool resolve_organism_from_json(const std::string& json_path,
                                        std::string& out_gtag,
                                        std::string& out_species,
                                        std::string& out_taxon) {
    FILE* fj = std::fopen(json_path.c_str(), "r");
    if (!fj) return false;
    std::fseek(fj, 0, SEEK_END);
    long sz = std::ftell(fj);
    std::fseek(fj, 0, SEEK_SET);
    if (sz <= 0 || sz >= 1000000) { std::fclose(fj); return false; }
    std::string meta(sz, '\0');
    std::fread(&meta[0], 1, sz, fj);
    std::fclose(fj);

    std::string organism  = json_extract_str(meta, "organism");
    std::string taxon_str = json_extract_str(meta, "taxon_id");
    // taxon_id may be a bare integer in the JSON (not quoted)
    if (taxon_str.empty()) {
        auto pos = meta.find("\"taxon_id\"");
        if (pos != std::string::npos) {
            pos = meta.find(':', pos);
            if (pos != std::string::npos) {
                ++pos;
                while (pos < meta.size() && (meta[pos]==' '||meta[pos]=='\t'||
                                              meta[pos]=='\n'||meta[pos]=='\r')) ++pos;
                if (pos < meta.size() && meta[pos] >= '1' && meta[pos] <= '9') {
                    while (pos < meta.size() && meta[pos] >= '0' && meta[pos] <= '9')
                        taxon_str += meta[pos++];
                }
            }
        }
    }

    // First: fast local map (~20 species)
    std::string gtag = organism_to_genome_tag(organism, taxon_str);

    // Second: full registry – covers all 37 registered assemblies via
    // case-insensitive substring match on scientific name.
    if (gtag.empty() && !organism.empty()) {
        namespace sreg = singlet::species_registry;
        const sreg::SpeciesInfo* sp = sreg::find_by_scientific(organism);
        if (!sp && !taxon_str.empty()) {
            try { sp = sreg::find_by_taxon(std::stoi(taxon_str)); } catch (...) {}
        }
        if (sp) {
            gtag      = sp->assembly;
            organism  = sp->scientific_name;
            taxon_str = std::to_string(sp->taxon_id);
        }
    }

    if (gtag.empty()) return false;
    out_gtag    = gtag;
    out_species = organism.empty() ? genome_tag_to_species(gtag) : organism;
    out_taxon   = taxon_str.empty() ? genome_tag_to_taxon(gtag)  : taxon_str;
    return true;
}

// 2-bit encoding table (N/other → 0xFF sentinel)
static const uint8_t BASE2BIT[256] = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,   0,0xFF,   1, 0xFF,0xFF,0xFF,   2, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,    3,   3,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,   0,0xFF,   1, 0xFF,0xFF,0xFF,   2, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,    3,   3,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    // remaining 128 entries → 0xFF
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
};

static constexpr uint64_t KMER_MASK = (1ULL << (2 * KMER_K)) - 1;

// Reverse complement of a 2-bit-encoded k-mer
inline uint64_t revcomp_kmer(uint64_t kmer) {
    // Complement bits: XOR with all-1s in pairs of 2
    uint64_t rc = ~kmer;
    // Reverse the pairs
    uint64_t out = 0;
    for (int i = 0; i < KMER_K; ++i) {
        out = (out << 2) | (rc & 3);
        rc >>= 2;
    }
    return out & KMER_MASK;
}

// Canonical k-mer = min(kmer, revcomp)
inline uint64_t canonical(uint64_t kmer) {
    uint64_t rc = revcomp_kmer(kmer);
    return kmer < rc ? kmer : rc;
}

// Extract and count canonical 21-mers from a byte-numeric sequence
// (STAR format: A=0, C=1, G=2, T=3)
inline int count_hits_numeric(const uint8_t* seq, uint16_t len,
                               const std::unordered_set<uint64_t>& db) {
    if (len < KMER_K) return 0;
    int hits = 0;
    // byte-numeric (0-3) not ASCII — need a different encoding table
    // STAR uses: A=0, C=1, G=2, T=3
    // canonical 21-mer computation same, just complementing is: 3^bit
    static const uint8_t COMP_NUM[4] = {3, 2, 1, 0};

    uint64_t kmer = 0;
    int valid = 0;
    for (int i = 0; i < len; ++i) {
        uint8_t b = seq[i];
        if (b > 3) { valid = 0; kmer = 0; continue; }
        kmer = ((kmer << 2) | b) & KMER_MASK;
        if (++valid >= KMER_K) {
            // Compute revcomp for numeric encoding
            uint64_t tmp = kmer;
            uint64_t rc = 0;
            for (int j = 0; j < KMER_K; ++j) {
                rc = (rc << 2) | COMP_NUM[tmp & 3];
                tmp >>= 2;
            }
            rc &= KMER_MASK;
            uint64_t canon = (kmer < rc) ? kmer : rc;
            if (db.count(canon)) ++hits;
        }
    }
    return hits;
}

} // namespace detail

// ── Public API ────────────────────────────────────────────────────────────────

/// Detect species from a .1fq file.
/// First tries metadata (instant), then falls back to k-mer sampling.
/// @param onefq_path       Path to the .1fq input file
/// @param verbose          Print diagnostic messages to stderr
/// @param extra_json_path  Optional: path to an external JSON file (e.g. the
///                         --metadata-json passed to singlify process) that may
///                         contain "organism" and/or "taxon_id" fields.  Checked
///                         as Path 1c — after .1fq metadata and sidecar but
///                         before expensive Bloom / k-mer sampling.  When
///                         k-mer detection returns 0 hits and this path is
///                         non-empty, a WARNING is emitted and the JSON organism
///                         is used directly.
inline DetectionResult detect(const std::string& onefq_path, bool verbose = true,
                               const std::string& extra_json_path = "") {
    DetectionResult result;
    result.method = "none";
    result.confidence = 0.0f;
    result.genome_tag = "unknown";

    // ── Path 1: Metadata ──────────────────────────────────────────────────────
    // Try .1fq internal metadata first, then sidecar metadata.json
    try {
        lib1fq::Reader probe;
        probe.open(onefq_path.c_str());
        const auto& hdr = probe.header();

        if (hdr.meta_size > 0) {
            try {
                std::string meta = probe.read_metadata();
                // Search for organism / taxon_id under sra_metadata block
                std::string organism  = detail::json_extract_str(meta, "organism");
                std::string taxon_id  = detail::json_extract_str(meta, "taxon_id");

                std::string gtag = detail::organism_to_genome_tag(organism, taxon_id);
                if (!gtag.empty()) {
                    result.genome_tag  = gtag;
                    result.species     = organism.empty()
                                         ? detail::genome_tag_to_species(gtag) : organism;
                    result.taxon_id    = taxon_id.empty()
                                         ? detail::genome_tag_to_taxon(gtag) : taxon_id;
                    result.confidence  = 1.0f;
                    result.method      = "metadata";
                    if (verbose) result.print();
                    return result;
                }
            } catch (...) {}
        }
    } catch (...) {}

    // ── Path 1b: Sidecar metadata.json ────────────────────────────────────────
    // Check for metadata.json in the same directory as the .1fq file
    // (created by val2_download.sh and other pipeline scripts)
    {
        std::string dir;
        auto slash = onefq_path.rfind('/');
        if (slash != std::string::npos) dir = onefq_path.substr(0, slash);
        else dir = ".";
        
        std::string sidecar = dir + "/metadata.json";
        FILE* mf = std::fopen(sidecar.c_str(), "r");
        if (mf) {
            // Read entire file
            std::fseek(mf, 0, SEEK_END);
            long sz = std::ftell(mf);
            std::fseek(mf, 0, SEEK_SET);
            if (sz > 0 && sz < 1000000) {
                std::string meta(sz, '\0');
                std::fread(&meta[0], 1, sz, mf);
                
                // Look for species_expected or organism field
                std::string species_exp = detail::json_extract_str(meta, "species_expected");
                std::string organism    = detail::json_extract_str(meta, "organism");
                std::string taxon_id    = detail::json_extract_str(meta, "taxon_id");
                
                // species_expected uses short names: human, mouse, chicken, etc.
                if (!species_exp.empty()) {
                    std::string gtag = detail::organism_to_genome_tag(species_exp, "");
                    if (!gtag.empty()) {
                        result.genome_tag  = gtag;
                        result.species     = detail::genome_tag_to_species(gtag);
                        result.taxon_id    = detail::genome_tag_to_taxon(gtag);
                        result.confidence  = 0.9f;  // slightly less than internal metadata
                        result.method      = "metadata";
                        std::fclose(mf);
                        if (verbose) {
                            std::fprintf(stderr, "[species-detect] sidecar metadata: %s → %s\n",
                                         species_exp.c_str(), gtag.c_str());
                            result.print();
                        }
                        return result;
                    }
                }
                // Fallback: try organism field
                if (!organism.empty() || !taxon_id.empty()) {
                    std::string gtag = detail::organism_to_genome_tag(organism, taxon_id);
                    if (!gtag.empty()) {
                        result.genome_tag  = gtag;
                        result.species     = organism.empty()
                                             ? detail::genome_tag_to_species(gtag) : organism;
                        result.taxon_id    = taxon_id.empty()
                                             ? detail::genome_tag_to_taxon(gtag) : taxon_id;
                        result.confidence  = 0.9f;
                        result.method      = "metadata";
                        std::fclose(mf);
                        if (verbose) result.print();
                        return result;
                    }
                }
            }
            std::fclose(mf);
        }
    }

    // ── Path 1c: Extra JSON (--metadata-json passed to singlify process) ──────
    // The job script writes organism/taxon_id to a JSON before calling download.
    // singlify download unfortunately overwrites the JSON with an empty skeleton,
    // losing the organism field. This path re-reads the --metadata-json explicitly
    // passed by the caller as a last-resort metadata source before expensive sampling.
    if (!extra_json_path.empty()) {
        FILE* ej = std::fopen(extra_json_path.c_str(), "r");
        if (ej) {
            std::fseek(ej, 0, SEEK_END);
            long sz = std::ftell(ej);
            std::fseek(ej, 0, SEEK_SET);
            if (sz > 0 && sz < 1000000) {
                std::string meta(sz, '\0');
                std::fread(&meta[0], 1, sz, ej);

                std::string organism = detail::json_extract_str(meta, "organism");
                std::string taxon_str = detail::json_extract_str(meta, "taxon_id");
                // taxon_id may be stored as a number; try extracting it as a string first,
                // then as a raw integer token if the string extraction returned nothing.
                if (taxon_str.empty()) {
                    auto pos = meta.find("\"taxon_id\"");
                    if (pos != std::string::npos) {
                        pos = meta.find(':', pos);
                        if (pos != std::string::npos) {
                            ++pos;
                            while (pos < meta.size() && (meta[pos] == ' ' || meta[pos] == '\t' ||
                                                          meta[pos] == '\n' || meta[pos] == '\r'))
                                ++pos;
                            // numeric token
                            if (pos < meta.size() && meta[pos] >= '1' && meta[pos] <= '9') {
                                while (pos < meta.size() && meta[pos] >= '0' && meta[pos] <= '9')
                                    taxon_str += meta[pos++];
                            }
                        }
                    }
                }

                std::string gtag = detail::organism_to_genome_tag(organism, taxon_str);
                // If not in local map, try species_registry for extended coverage (all 37 species)
                if (gtag.empty() && !organism.empty()) {
                    namespace sreg = singlet::species_registry;
                    const sreg::SpeciesInfo* sp = sreg::find_by_scientific(organism);
                    if (!sp && !taxon_str.empty()) {
                        try { sp = sreg::find_by_taxon(std::stoi(taxon_str)); } catch (...) {}
                    }
                    if (sp) {
                        gtag      = sp->assembly;
                        organism  = sp->scientific_name;
                        taxon_str = std::to_string(sp->taxon_id);
                    }
                }
                if (!gtag.empty()) {
                    result.genome_tag = gtag;
                    result.species    = organism.empty()
                                        ? detail::genome_tag_to_species(gtag) : organism;
                    result.taxon_id   = taxon_str.empty()
                                        ? detail::genome_tag_to_taxon(gtag) : taxon_str;
                    result.confidence = 0.95f;
                    result.method     = "metadata";
                    std::fclose(ej);
                    if (verbose) {
                        std::fprintf(stderr,
                            "[species-detect] WARNING: k-mer detection unavailable; "
                            "falling back to --metadata-json organism: %s → %s\n",
                            organism.empty() ? "(empty)" : organism.c_str(), gtag.c_str());
                        result.print();
                    }
                    return result;
                }
            }
            std::fclose(ej);
        }
    }

    // ── Path 2: Bloom filter scoring ──────────────────────────────────────────
    // Try Bloom filter files first (comprehensive transcriptome coverage).
    // Bloom filter location: ${SINGLIFY_SPECIES_FILTERS} env var, or
    // species_filters/ relative to the binary's source directory.
    {
        // Candidate directories to search for *.bloom files
        std::vector<std::string> bloom_dirs;
        if (const char* env = std::getenv("SINGLIFY_SPECIES_FILTERS"))
            bloom_dirs.push_back(env);
        // Add path relative to this header's install location
        // (determined at build time via __FILE__ — strip filename)
        {
            std::string hpath = __FILE__;  // .../include/singlet-pileup/species_detect.h
            auto slash = hpath.rfind('/');
            if (slash != std::string::npos) {
                std::string base = hpath.substr(0, slash); // .../include/singlet-pileup
                auto slash2 = base.rfind('/');
                if (slash2 != std::string::npos)
                    base = base.substr(0, slash2); // .../include
                slash2 = base.rfind('/');
                if (slash2 != std::string::npos)
                    base = base.substr(0, slash2); // singlify root
                bloom_dirs.push_back(base + "/species_filters");
            }
        }
        // Try each candidate dir
        for (const auto& dir : bloom_dirs) {
            std::string h_path = dir + "/human_21mer.bloom";
            std::string m_path = dir + "/mouse_21mer.bloom";
            singlet_pileup::BloomFilter bf_human, bf_mouse;
            if (!bf_human.load(h_path) || !bf_mouse.load(m_path))
                continue; // try next dir or fall through to diagnostic k-mers

            if (verbose)
                std::fprintf(stderr, "[species-detect] Bloom filters loaded from %s\n", dir.c_str());

            int human_hits = 0, mouse_hits = 0, total_kmers = 0;
            try {
                lib1fq::Reader reader;
                reader.open(onefq_path.c_str());
                const int arc_clip = (reader.header().protocol_id == 22) ? 50 : 0;
                // Detect cDNA stream: in a correctly-encoded .1fq R1=barcode(short),
                // R2=cDNA(long). If the encoder failed to swap orientations (e.g. BD
                // Rhapsody or SPLiT-seq VDB deposits where R1=cDNA, R2=barcode), the
                // header stream_lengths reveal this: R1 > 2×R2 means R1 is the cDNA.
                const uint16_t sl0_bloom = reader.header().stream_lengths[0];
                const uint16_t sl1_bloom = reader.header().stream_lengths[1];
                const bool use_r1_bloom = (sl0_bloom > 0 && sl1_bloom > 0 &&
                                           sl0_bloom > 2 * sl1_bloom);
                if (use_r1_bloom && verbose)
                    std::fprintf(stderr,
                        "[species-detect] R1(%dbp) > 2\xc3\x97R2(%dbp): "
                        "sampling R1 as cDNA stream for k-mer detection\n",
                        sl0_bloom, sl1_bloom);
                lib1fq::DecodedBlock blk;
                int reads_done = 0;

                while (reader.read_block(blk) && reads_done < SAMPLE_READS) {
                    for (uint32_t i = 0; i < blk.n_reads && reads_done < SAMPLE_READS; ++i) {
                        const uint8_t* seq = use_r1_bloom ? blk.r1_seq(i) : blk.r2_seq(i);
                        uint16_t len = use_r1_bloom ? blk.r1_len(i) : blk.r2_len(i);
                        if (arc_clip > 0 && len > static_cast<uint16_t>(arc_clip)) {
                            seq += arc_clip;
                            len  = static_cast<uint16_t>(len - arc_clip);
                        }
                        if (len < static_cast<uint16_t>(KMER_K)) continue;
                        human_hits  += bf_human.count_hits_numeric(seq, len, KMER_K);
                        mouse_hits  += bf_mouse.count_hits_numeric(seq, len, KMER_K);
                        total_kmers += std::max(0, (int)len - KMER_K + 1);
                        ++reads_done;
                    }
                }

                result.reads_sampled = reads_done;
                result.kmers_tested  = total_kmers;

                if (total_kmers > 0) {
                    float h_rate = static_cast<float>(human_hits) / total_kmers;
                    float m_rate = static_cast<float>(mouse_hits) / total_kmers;
                    float h_vs_m = h_rate / (m_rate + 1e-9f);
                    float m_vs_h = m_rate / (h_rate + 1e-9f);

                    if (h_rate >= CONFIDENCE_THRESH && h_vs_m >= MIN_RATIO && h_rate > m_rate) {
                        result.genome_tag  = "GRCh38";
                        result.species     = "Homo sapiens";
                        result.taxon_id    = "9606";
                        result.confidence  = h_rate / (h_rate + m_rate + 1e-9f);
                        result.kmers_hit   = human_hits;
                        result.method      = "bloom";
                    } else if (m_rate >= CONFIDENCE_THRESH && m_vs_h >= MIN_RATIO && m_rate > h_rate) {
                        result.genome_tag  = "GRCm39";
                        result.species     = "Mus musculus";
                        result.taxon_id    = "10090";
                        result.confidence  = m_rate / (h_rate + m_rate + 1e-9f);
                        result.kmers_hit   = mouse_hits;
                        result.method      = "bloom";
                    } else {
                        result.method      = "bloom";
                        result.confidence  = 0.0f;
                        result.kmers_hit   = std::max(human_hits, mouse_hits);
                    }
                }
            } catch (...) {}
            // Metadata organism fallback: when Bloom returned no confident call,
            // try --metadata-json organism via species_registry before giving up.
            if (result.confidence == 0.0f && !extra_json_path.empty()) {
                std::string fb_gtag, fb_species, fb_taxon;
                if (detail::resolve_organism_from_json(extra_json_path, fb_gtag, fb_species, fb_taxon)) {
                    if (verbose)
                        std::fprintf(stderr,
                            "[species_detect] k-mer detection returned 0/%d hits; "
                            "falling back to metadata organism: %s (genome_tag=%s)\n",
                            result.kmers_hit, fb_species.c_str(), fb_gtag.c_str());
                    result.genome_tag = fb_gtag;
                    result.species    = fb_species;
                    result.taxon_id   = fb_taxon;
                    result.confidence = 0.10f;
                    result.method     = "metadata_fallback";
                }
            }
            if (verbose) result.print();
            return result; // Bloom filter path succeeded (even if inconclusive)
        }
    }

    // ── Path 3: Diagnostic k-mer fallback (no Bloom filter files found) ─────
#if SINGLET_HAVE_KMER_DB
    // Build one unordered_set per species (done once, static init at first call).
    struct SpeciesDb {
        std::string genome_tag;
        std::string species;
        std::string taxon_id;
        std::unordered_set<uint64_t> db;
    };
    static const std::vector<SpeciesDb> s_all_dbs = [] {
        std::vector<SpeciesDb> v;
        v.reserve(species_kmer_db::ALL_SPECIES_COUNT);
        for (size_t i = 0; i < species_kmer_db::ALL_SPECIES_COUNT; ++i) {
            const auto& e = species_kmer_db::ALL_SPECIES[i];
            SpeciesDb s;
            s.genome_tag = e.genome_tag;
            s.species    = e.species_name;
            s.taxon_id   = e.taxon_id;
            s.db = std::unordered_set<uint64_t>(e.kmers, e.kmers + e.count);
            v.push_back(std::move(s));
        }
        return v;
    }();

    const size_t n_sp = s_all_dbs.size();
    std::vector<int> hits(n_sp, 0);
    int total_kmers = 0;

    try {
        lib1fq::Reader reader;
        reader.open(onefq_path.c_str());
        // 10x-arc-gex (protocol_id=22): R2 has a 50bp constant ARC ATAC linker
        // at the 5' end before the cDNA. Skip it to avoid k-mer misses.
        const int arc_clip = (reader.header().protocol_id == 22) ? 50 : 0;
        // Detect cDNA stream: if R1 fixed length > 2×R2 fixed length, the encoder
        // stored cDNA in R1 (read orientation not swapped). Sample R1 in that case.
        const uint16_t sl0_kmer = reader.header().stream_lengths[0];
        const uint16_t sl1_kmer = reader.header().stream_lengths[1];
        const bool use_r1_kmer = (sl0_kmer > 0 && sl1_kmer > 0 &&
                                   sl0_kmer > 2 * sl1_kmer);
        if (use_r1_kmer && verbose)
            std::fprintf(stderr,
                "[species-detect] R1(%dbp) > 2\xc3\x97R2(%dbp): "
                "sampling R1 as cDNA stream for k-mer detection\n",
                sl0_kmer, sl1_kmer);
        lib1fq::DecodedBlock blk;
        int reads_done = 0;

        while (reader.read_block(blk) && reads_done < SAMPLE_READS) {
            for (uint32_t i = 0; i < blk.n_reads && reads_done < SAMPLE_READS; ++i) {
                const uint8_t* seq = use_r1_kmer ? blk.r1_seq(i) : blk.r2_seq(i);
                uint16_t len = use_r1_kmer ? blk.r1_len(i) : blk.r2_len(i);
                if (arc_clip > 0 && len > static_cast<uint16_t>(arc_clip)) {
                    seq += arc_clip;
                    len  = static_cast<uint16_t>(len - arc_clip);
                }
                if (len < static_cast<uint16_t>(KMER_K)) continue;
                for (size_t si = 0; si < n_sp; ++si)
                    hits[si] += detail::count_hits_numeric(seq, len, s_all_dbs[si].db);
                total_kmers += std::max(0, (int)len - KMER_K + 1);
                ++reads_done;
            }
        }

        result.reads_sampled = reads_done;
        result.kmers_tested  = total_kmers;

        if (total_kmers > 0) {
            // Find best-scoring species
            int best_idx = 0;
            for (size_t i = 1; i < n_sp; ++i)
                if (hits[i] > hits[best_idx]) best_idx = static_cast<int>(i);

            // Find second-best
            int second = 0;
            for (size_t i = 0; i < n_sp; ++i)
                if (static_cast<int>(i) != best_idx && hits[i] > second)
                    second = hits[i];

            float best_rate = static_cast<float>(hits[best_idx]) / total_kmers;
            float ratio     = static_cast<float>(hits[best_idx]) / (second + 1e-9f);

            if (best_rate >= CONFIDENCE_THRESH && ratio >= MIN_RATIO) {
                result.genome_tag = s_all_dbs[best_idx].genome_tag;
                result.species    = s_all_dbs[best_idx].species;
                result.taxon_id   = s_all_dbs[best_idx].taxon_id;
                float total_hit_rate = static_cast<float>(hits[best_idx] + second) / total_kmers;
                result.confidence = (total_hit_rate > 0)
                    ? best_rate / (total_hit_rate + 1e-9f)
                    : best_rate;
                result.kmers_hit  = hits[best_idx];
                result.method     = "kmer";
            } else {
                result.method     = "kmer";
                result.confidence = 0.0f;
                result.kmers_hit  = hits[best_idx];
            }
        }
    } catch (...) {}

    // Metadata organism fallback: when k-mer sampling returned no confident call,
    // try --metadata-json organism via species_registry before giving up.
    if (result.confidence == 0.0f && !extra_json_path.empty()) {
        std::string fb_gtag, fb_species, fb_taxon;
        if (detail::resolve_organism_from_json(extra_json_path, fb_gtag, fb_species, fb_taxon)) {
            if (verbose)
                std::fprintf(stderr,
                    "[species_detect] k-mer detection returned 0/%d hits; "
                    "falling back to metadata organism: %s (genome_tag=%s)\n",
                    result.kmers_hit, fb_species.c_str(), fb_gtag.c_str());
            result.genome_tag = fb_gtag;
            result.species    = fb_species;
            result.taxon_id   = fb_taxon;
            result.confidence = 0.10f;
            result.method     = "metadata_fallback";
        }
    }

    if (verbose) result.print();
    return result;
#else
    // No k-mer DB — fall back to metadata organism if available
    if (!extra_json_path.empty()) {
        std::string fb_gtag, fb_species, fb_taxon;
        if (detail::resolve_organism_from_json(extra_json_path, fb_gtag, fb_species, fb_taxon)) {
            if (verbose)
                std::fprintf(stderr,
                    "[species_detect] No k-mer DB; falling back to metadata organism: %s (genome_tag=%s)\n",
                    fb_species.c_str(), fb_gtag.c_str());
            result.genome_tag = fb_gtag;
            result.species    = fb_species;
            result.taxon_id   = fb_taxon;
            result.confidence = 0.10f;
            result.method     = "metadata_fallback";
            if (verbose) result.print();
            return result;
        }
    }
    // No k-mer DB and no usable metadata
    if (verbose)
        std::fprintf(stderr,
            "[species-detect] No k-mer DB (run scripts/build_species_kmer_db.py). "
            "Metadata-only detection: no organism found.\n");
    return result;
#endif
}

/// Quick organism string from .1fq metadata only (no k-mer sampling).
/// Returns empty string if not found.
inline std::string organism_from_metadata(const std::string& onefq_path) {
    try {
        lib1fq::Reader probe;
        probe.open(onefq_path.c_str());
        if (probe.header().meta_size == 0) return "";
        std::string meta = probe.read_metadata();
        return detail::json_extract_str(meta, "organism");
    } catch (...) {
        return "";
    }
}

/// Parse a STAR genomeParameters.txt to extract the genome assembly tag.
/// Returns "GRCh38", "GRCm39", or "" if not recognised.
inline std::string genome_tag_from_star_dir(const std::string& genome_dir) {
    std::string params_path = genome_dir + "/genomeParameters.txt";
    FILE* f = std::fopen(params_path.c_str(), "r");
    if (!f) return "";

    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        std::string s(line);
        // Look for genomeFastaFiles line which contains the FASTA path
        if (s.find("genomeFastaFiles") != std::string::npos) {
            std::fclose(f);
            // Check path for known assembly identifiers
            auto lo = s;
            std::transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
            if (lo.find("grch38") != std::string::npos) return "GRCh38";
            if (lo.find("grcm39") != std::string::npos) return "GRCm39";
            if (lo.find("hg38")   != std::string::npos) return "GRCh38";
            if (lo.find("mm39")   != std::string::npos) return "GRCm39";
            return "";
        }
    }
    std::fclose(f);
    return "";
}

} // namespace species_detect
