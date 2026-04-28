// lib1fq/manifest.h — Library bundle manifest
//
// A bundle ties multiple .1fq files from the same experiment together,
// declaring the biological role of each library and the shared cell
// barcode namespace. This allows multiome (GEX+ATAC), CITE-seq (GEX+ADT),
// and HTO-demultiplexed datasets to be represented as a coherent unit.
//
// Bundle manifest is written as a JSON sidecar file alongside the .1fq files.
// Canonical filename: <experiment_accession>.bundle.json
// (or <base>.1fq.bundle.json when co-located with a primary library)

#pragma once

#include "types.h"
#include "protocol.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace lib1fq {

// ── Library entry within a bundle ──

struct BundleLibrary {
    // Role of this library within the experiment
    // e.g. "gex", "atac", "adt", "hto", "crispr", "vdj", "spatial"
    std::string role;

    // Path to the .1fq file (relative to manifest, or absolute)
    std::string file;

    // Assay type for this library
    AssayType assay_type = AssayType::UNKNOWN;

    // Protocol detected for this library
    std::string protocol_tag;

    // Canonical protocol URI (from protocol_uri())
    std::string protocol_uri_str;

    // Run accession (SRR / ERR / DRR)
    std::string srr;

    // EFO ontology term
    std::string efo_id;
    std::string efo_label;

    // Number of reads in this library
    uint64_t total_reads = 0;
};

// ── Bundle manifest ──

struct BundleManifest {
    // Manifest format identifier
    static constexpr const char* FORMAT = "1fq-bundle";
    static constexpr const char* FORMAT_VERSION = "1.0";

    // Experiment-level identifiers (from SRA/GEO)
    std::string bundle_id;            // Unique ID, typically the SRX accession
    std::string experiment_accession; // SRX / ERX / DRX
    std::string study_accession;      // SRP / ERP / DRP
    std::string gse;                  // GEO series
    std::string biosample;            // BioSample accession

    // Organism context
    std::string organism;
    std::string taxon_id;

    // Shared barcode namespace: true if all libraries share the same
    // cell barcode whitelist (common in multiome / CITE-seq).
    bool shared_barcode_namespace = false;

    // The libraries in this bundle, in processing order
    std::vector<BundleLibrary> libraries;

    // ── Write manifest to file ──
    // Path should end in .bundle.json by convention.
    void write(const std::string& path) const {
        FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp)
            throw std::runtime_error("Cannot open bundle manifest for writing: " + path);

        std::string j = to_json();
        if (std::fwrite(j.data(), 1, j.size(), fp) != j.size()) {
            std::fclose(fp);
            throw std::runtime_error("Failed to write bundle manifest: " + path);
        }
        std::fclose(fp);
    }

    std::string to_json() const {
        std::string j;
        j.reserve(2048);
        j += "{\n";
        j += "  \"format\": \"" + std::string(FORMAT) + "\",\n";
        j += "  \"version\": \"" + std::string(FORMAT_VERSION) + "\",\n";
        j += "  \"bundle_id\": " + js(bundle_id) + ",\n";
        j += "  \"experiment_accession\": " + js(experiment_accession) + ",\n";
        j += "  \"study_accession\": " + js(study_accession) + ",\n";
        j += "  \"gse\": " + js(gse) + ",\n";
        j += "  \"biosample\": " + js(biosample) + ",\n";
        j += "  \"organism\": " + js(organism) + ",\n";
        j += "  \"taxon_id\": " + js(taxon_id) + ",\n";
        j += "  \"shared_barcode_namespace\": ";
        j += (shared_barcode_namespace ? "true" : "false");
        j += ",\n";
        j += "  \"libraries\": [\n";
        for (size_t i = 0; i < libraries.size(); ++i) {
            const auto& lib = libraries[i];
            j += "    {\n";
            j += "      \"role\": " + js(lib.role) + ",\n";
            j += "      \"file\": " + js(lib.file) + ",\n";
            j += "      \"srr\": " + js(lib.srr) + ",\n";
            j += "      \"assay_type\": " + js(assay_type_name(lib.assay_type)) + ",\n";
            j += "      \"protocol\": " + js(lib.protocol_tag) + ",\n";
            if (!lib.protocol_uri_str.empty())
                j += "      \"protocol_uri\": " + js(lib.protocol_uri_str) + ",\n";
            else
                j += "      \"protocol_uri\": null,\n";
            if (!lib.efo_id.empty()) {
                j += "      \"efo_id\": " + js(lib.efo_id) + ",\n";
                j += "      \"efo_label\": " + js(lib.efo_label) + ",\n";
            } else {
                j += "      \"efo_id\": null,\n";
                j += "      \"efo_label\": null,\n";
            }
            j += "      \"total_reads\": " + std::to_string(lib.total_reads) + "\n";
            j += "    }";
            if (i + 1 < libraries.size()) j += ",";
            j += "\n";
        }
        j += "  ]\n";
        j += "}\n";
        return j;
    }

private:
    // Minimal JSON string escaper
    static std::string js(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 4);
        out += '"';
        for (unsigned char c : s) {
            if      (c == '"')  { out += "\\\""; }
            else if (c == '\\') { out += "\\\\"; }
            else if (c == '\n') { out += "\\n";  }
            else if (c == '\r') { out += "\\r";  }
            else if (c == '\t') { out += "\\t";  }
            else if (c < 0x20)  { /* skip control chars */ }
            else                { out += static_cast<char>(c); }
        }
        out += '"';
        return out;
    }
};

// ── Convenience: build a single-library bundle from encoder output ──
// Call after encoding to produce a minimal bundle.json for the run.

inline BundleManifest make_single_library_bundle(
        const std::string& file_path,
        const std::string& srr,
        const std::string& srx,
        const std::string& srp,
        const std::string& gse,
        const std::string& biosample,
        const std::string& organism,
        const std::string& taxon_id,
        const std::string& role,
        AssayType assay_type,
        const std::string& protocol_tag,
        uint64_t total_reads) {

    BundleManifest manifest;
    manifest.bundle_id            = srx;
    manifest.experiment_accession = srx;
    manifest.study_accession      = srp;
    manifest.gse                  = gse;
    manifest.biosample             = biosample;
    manifest.organism             = organism;
    manifest.taxon_id             = taxon_id;
    manifest.shared_barcode_namespace = false;

    BundleLibrary lib;
    lib.role             = role;
    lib.file             = file_path;
    lib.srr              = srr;
    lib.assay_type       = assay_type;
    lib.protocol_tag     = protocol_tag;
    lib.protocol_uri_str = protocol_uri(protocol_tag);
    lib.total_reads      = total_reads;

    auto efo = assay_efo_term(assay_type);
    lib.efo_id    = efo.id    ? efo.id    : "";
    lib.efo_label = efo.label ? efo.label : "";

    manifest.libraries.push_back(std::move(lib));
    return manifest;
}

} // namespace lib1fq
