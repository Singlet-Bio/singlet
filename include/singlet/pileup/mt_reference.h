// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: mt_reference.h
// Extract and write the mitochondrial reference genome from STAR's Genome binary.
// Reads chrName.txt, chrStart.txt, chrNameLength.txt from the genome directory,
// locates chrM (or MT/chrMT), and writes it as a FASTA file.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace singlet {

/// Result of MT reference extraction.
struct MtReferenceResult {
    bool        found = false;
    std::string contig_name;     ///< "chrM", "MT", or "chrMT"
    uint32_t    length = 0;
    std::string sequence;        ///< ACGTN sequence
    std::string reference_build; ///< e.g. "GRCh38-2024-A"
};

/// Extract the mitochondrial reference sequence from a STAR genome directory.
///
/// Reads Genome binary + metadata files to locate and extract chrM.
/// Supports contig names: chrM, MT, chrMT (case-sensitive).
///
/// @param genome_dir  Path to STAR genome directory (containing Genome, chrName.txt, etc.)
/// @param ref_build   Reference build name for FASTA header
/// @return MtReferenceResult with sequence if found
inline MtReferenceResult extract_mt_reference(const std::string& genome_dir,
                                               const std::string& ref_build = "") {
    MtReferenceResult result;
    result.reference_build = ref_build;

    // Read chromosome names
    std::vector<std::string> chr_names;
    {
        std::ifstream f(genome_dir + "/chrName.txt");
        if (!f) return result;
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) chr_names.push_back(line);
        }
    }

    // Read chromosome start offsets
    std::vector<uint64_t> chr_starts;
    {
        std::ifstream f(genome_dir + "/chrStart.txt");
        if (!f) return result;
        uint64_t v;
        while (f >> v) chr_starts.push_back(v);
    }

    // Read chromosome lengths
    std::vector<uint64_t> chr_lengths;
    {
        std::ifstream f(genome_dir + "/chrNameLength.txt");
        if (!f) return result;
        std::string name;
        uint64_t len;
        while (f >> name >> len) chr_lengths.push_back(len);
    }

    if (chr_names.size() != chr_lengths.size()) return result;
    if (chr_starts.size() < chr_names.size()) return result;

    // Find chrM / MT / chrMT
    size_t mt_idx = SIZE_MAX;
    for (size_t i = 0; i < chr_names.size(); ++i) {
        if (chr_names[i] == "chrM" || chr_names[i] == "MT" || chr_names[i] == "chrMT") {
            mt_idx = i;
            break;
        }
    }
    if (mt_idx == SIZE_MAX) return result;

    result.contig_name = chr_names[mt_idx];
    result.length = static_cast<uint32_t>(chr_lengths[mt_idx]);

    // Read sequence from Genome binary
    // STAR stores 1 byte per base: 0=A, 1=C, 2=G, 3=T, 4+=N
    std::ifstream gf(genome_dir + "/Genome", std::ios::binary);
    if (!gf) return result;

    gf.seekg(static_cast<std::streamoff>(chr_starts[mt_idx]));
    std::vector<uint8_t> raw(result.length);
    gf.read(reinterpret_cast<char*>(raw.data()), result.length);
    if (!gf) return result;

    static const char BASE_MAP[] = "ACGTN";
    result.sequence.resize(result.length);
    for (uint32_t i = 0; i < result.length; ++i) {
        result.sequence[i] = (raw[i] < 4) ? BASE_MAP[raw[i]] : 'N';
    }

    result.found = true;
    return result;
}

/// Write the MT reference as a FASTA file.
///
/// @param filepath  Output path (e.g., "output/mt_reference.fa")
/// @param mt_ref    Extracted MT reference
/// @return true on success
inline bool write_mt_reference_fasta(const std::string& filepath,
                                      const MtReferenceResult& mt_ref) {
    std::ofstream f(filepath + ".tmp");
    if (!f) return false;

    // FASTA header
    f << ">" << mt_ref.contig_name;
    if (!mt_ref.reference_build.empty())
        f << " " << mt_ref.reference_build;
    f << " " << mt_ref.length << "bp\n";

    // Sequence in 80-char lines
    for (uint32_t i = 0; i < mt_ref.length; i += 80) {
        uint32_t end = std::min(i + 80, mt_ref.length);
        f.write(mt_ref.sequence.data() + i, end - i);
        f << '\n';
    }

    f.close();
    if (!f.good()) {
        std::error_code ec;
        std::filesystem::remove(filepath + ".tmp", ec);
        return false;
    }
    // Atomic rename
    std::error_code ec;
    std::filesystem::rename(filepath + ".tmp", filepath, ec);
    return !ec;
}

/// Write an empty MT reference stub when chrM is not found in the genome.
inline bool write_mt_reference_stub(const std::string& filepath,
                                     const std::string& reason) {
    std::ofstream f(filepath + ".tmp");
    if (!f) return false;
    f << ">chrM_not_found " << reason << "\n";
    f.close();
    std::error_code ec;
    std::filesystem::rename(filepath + ".tmp", filepath, ec);
    return !ec;
}

}  // namespace singlet
