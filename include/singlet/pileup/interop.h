#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace singlet_pileup::interop {

// ── Output Format Validators ──

struct ValidationResult {
    bool valid;
    std::string error;  // empty if valid
};

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace detail {

inline bool file_exists(const std::string& p) {
    return std::filesystem::exists(p);
}

inline bool file_exists_any(const std::string& base,
                            const std::vector<std::string>& suffixes,
                            std::string& found) {
    for (auto& s : suffixes) {
        std::string p = base + s;
        if (std::filesystem::exists(p)) {
            found = p;
            return true;
        }
    }
    return false;
}

// Read first line of a (plain-text) file
inline std::string first_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    return line;
}

// Count tab-separated columns in a line
inline int count_cols(const std::string& line, char sep = '\t') {
    if (line.empty()) return 0;
    int n = 1;
    for (char c : line)
        if (c == sep) ++n;
    return n;
}

}  // namespace detail

// ── Validators ────────────────────────────────────────────────────────────────

// Validate CellRanger-compatible MTX directory structure.
// Accepts both compressed (.gz) and uncompressed variants.
inline ValidationResult validate_cellranger_mtx(const std::string& dir_path) {
    if (!std::filesystem::is_directory(dir_path))
        return {false, "directory does not exist: " + dir_path};

    // Check for matrix file
    std::string matrix_path;
    if (!detail::file_exists_any(dir_path + "/matrix.mtx", {"", ".gz"}, matrix_path))
        return {false, "missing matrix.mtx or matrix.mtx.gz in " + dir_path};

    // Check for features file
    std::string features_path;
    if (!detail::file_exists_any(dir_path + "/features.tsv", {"", ".gz"}, features_path))
        return {false, "missing features.tsv or features.tsv.gz in " + dir_path};

    // Check for barcodes file
    std::string barcodes_path;
    if (!detail::file_exists_any(dir_path + "/barcodes.tsv", {"", ".gz"}, barcodes_path))
        return {false, "missing barcodes.tsv or barcodes.tsv.gz in " + dir_path};

    // For uncompressed files: validate content
    if (matrix_path == dir_path + "/matrix.mtx") {
        std::string hdr = detail::first_line(matrix_path);
        if (hdr.find("%%MatrixMarket") == std::string::npos)
            return {false, "matrix.mtx missing %%MatrixMarket header"};
    }

    if (features_path == dir_path + "/features.tsv") {
        std::string line;
        std::ifstream f(features_path);
        if (std::getline(f, line)) {
            int cols = detail::count_cols(line, '\t');
            // features.tsv can have 2 or 3 columns (gene_id, gene_name [, feature_type])
            if (cols < 2)
                return {false, "features.tsv must have ≥2 tab-delimited columns"};
        }
    }

    if (barcodes_path == dir_path + "/barcodes.tsv") {
        std::string line;
        std::ifstream f(barcodes_path);
        if (std::getline(f, line)) {
            if (line.find("-1") == std::string::npos && line.find("-") == std::string::npos)
                return {false, "barcodes.tsv: first barcode missing '-1' suffix: " + line};
        }
    }

    return {true, ""};
}

// Validate fragments.tsv for ArchR/Signac compatibility.
// Expected columns (tab-separated): chr, start, end, barcode, count
inline ValidationResult validate_fragments_tsv(const std::string& file_path) {
    if (!detail::file_exists(file_path))
        return {false, "file does not exist: " + file_path};

    std::ifstream f(file_path);
    std::string line;
    int lineno = 0;
    std::string prev_chr;

    while (std::getline(f, line)) {
        // Skip comment lines
        if (!line.empty() && line[0] == '#') continue;
        ++lineno;

        std::istringstream ss(line);
        std::string chr, start_s, end_s, barcode, count_s;
        if (!std::getline(ss, chr, '\t') || !std::getline(ss, start_s, '\t') || !std::getline(ss, end_s, '\t') || !std::getline(ss, barcode, '\t') || !std::getline(ss, count_s, '\t')) {
            std::string extra;
            // 5 columns minimum; count_s may be last with no trailing tab
            if (!std::getline(ss, count_s)) {
                // try reading count_s as 5th field with no trailing newline
            }
            if (count_s.empty())
                return {false, "fragments.tsv line " + std::to_string(lineno) + ": expected 5 tab-delimited columns"};
        }

        // Parse start/end as integers
        int64_t start_v, end_v;
        try {
            start_v = std::stoll(start_s);
            end_v = std::stoll(end_s);
        } catch (...) {
            return {false, "fragments.tsv line " + std::to_string(lineno) + ": non-numeric coordinates"};
        }

        if (start_v >= end_v)
            return {false, "fragments.tsv line " + std::to_string(lineno) + ": start >= end (" + start_s + " >= " + end_s + ")"};

        if (lineno > 1) break;  // only validate first data line for content check
    }

    if (lineno == 0)
        return {false, "fragments.tsv: no data lines found"};

    return {true, ""};
}

// Validate spatial_coordinates.tsv for Seurat Load10X_Spatial.
// Expected: barcode, in_tissue, array_row, array_col [, pxl_row_in_fullres, pxl_col_in_fullres]
inline ValidationResult validate_spatial_coords(const std::string& file_path) {
    if (!detail::file_exists(file_path))
        return {false, "file does not exist: " + file_path};

    std::ifstream f(file_path);
    std::string line;
    if (!std::getline(f, line))
        return {false, "spatial_coordinates.tsv: empty file"};

    int cols = detail::count_cols(line, '\t');
    if (cols < 4)
        return {false, "spatial_coordinates.tsv: expected ≥4 columns, got " + std::to_string(cols)};

    return {true, ""};
}

// Validate .pairs file for 4DN Hi-C.
// Header must start with "## pairs format v1.0"
inline ValidationResult validate_4dn_pairs(const std::string& file_path) {
    if (!detail::file_exists(file_path))
        return {false, "file does not exist: " + file_path};

    std::string hdr = detail::first_line(file_path);
    if (hdr.find("## pairs format") == std::string::npos)
        return {false, ".pairs file missing '## pairs format' header line"};

    // Find #columns header or first data line
    std::ifstream f(file_path);
    std::string line;
    bool found_columns = false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] != '#') {
            // data line — check has ≥7 tab-separated fields
            int cols = detail::count_cols(line, '\t');
            if (cols < 7)
                return {false, ".pairs data line has fewer than 7 columns"};
            break;
        }
        if (line.find("#columns") != std::string::npos)
            found_columns = true;
    }
    (void)found_columns;

    return {true, ""};
}

// Validate Bismark-compatible CpG BED.
// Columns: chr, start, end, methylation_pct, count_meth, count_unmeth
inline ValidationResult validate_bismark_bed(const std::string& file_path) {
    if (!detail::file_exists(file_path))
        return {false, "file does not exist: " + file_path};

    std::ifstream f(file_path);
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '#') continue;
        ++lineno;

        std::istringstream ss(line);
        std::string chr, start_s, end_s, meth_s, cnt_m, cnt_u;
        if (!std::getline(ss, chr, '\t') || !std::getline(ss, start_s, '\t') || !std::getline(ss, end_s, '\t') || !std::getline(ss, meth_s, '\t') || !std::getline(ss, cnt_m, '\t') || !std::getline(ss, cnt_u, '\t'))
            return {false, "bismark BED line " + std::to_string(lineno) + ": expected 6 tab-delimited columns"};

        double meth;
        try {
            meth = std::stod(meth_s);
        } catch (...) {
            return {false, "bismark BED line " + std::to_string(lineno) + ": non-numeric methylation value"};
        }
        if (meth < 0.0 || meth > 100.0)
            return {false, "bismark BED line " + std::to_string(lineno) + ": methylation out of range [0,100]: " + meth_s};

        if (lineno >= 3) break;  // validate first few data lines
    }

    if (lineno == 0)
        return {false, "bismark BED: no data lines"};

    return {true, ""};
}

// Validate metrics_summary.csv for CellRanger compatibility.
inline ValidationResult validate_metrics_csv(const std::string& file_path) {
    if (!detail::file_exists(file_path))
        return {false, "file does not exist: " + file_path};

    std::ifstream f(file_path);
    std::string header;
    if (!std::getline(f, header))
        return {false, "metrics_summary.csv: empty file"};

    // Must be comma-separated
    if (header.find(',') == std::string::npos)
        return {false, "metrics_summary.csv: header is not comma-separated"};

    // Must contain key CellRanger field
    if (header.find("Estimated Number of Cells") == std::string::npos)
        return {false, "metrics_summary.csv: missing 'Estimated Number of Cells' field"};

    return {true, ""};
}

// Validate VCF output.
inline ValidationResult validate_vcf(const std::string& file_path) {
    if (!detail::file_exists(file_path))
        return {false, "file does not exist: " + file_path};

    std::ifstream f(file_path);
    std::string line;

    // First line must be ##fileformat=VCFv4.x
    if (!std::getline(f, line))
        return {false, "VCF: empty file"};
    if (line.find("##fileformat=VCF") == std::string::npos)
        return {false, "VCF: missing ##fileformat=VCFv4.x header"};

    // Find #CHROM line
    bool found_chrom = false;
    while (std::getline(f, line)) {
        if (line.rfind("#CHROM", 0) == 0) {
            found_chrom = true;
            int cols = detail::count_cols(line, '\t');
            if (cols < 8)
                return {false, "VCF #CHROM line has fewer than 8 columns"};
            break;
        }
    }
    if (!found_chrom)
        return {false, "VCF: missing #CHROM header line"};

    return {true, ""};
}

// ── Format Generators for Testing ────────────────────────────────────────────

// Generate synthetic CellRanger MTX directory with known data.
inline std::string generate_test_mtx(const std::string& output_dir,
                                     int n_genes, int n_cells, int n_entries) {
    std::filesystem::create_directories(output_dir);

    // matrix.mtx
    {
        std::ofstream f(output_dir + "/matrix.mtx");
        f << "%%MatrixMarket matrix coordinate integer general\n";
        f << "%\n";
        int actual = std::min(n_entries, n_genes * n_cells);
        f << n_genes << " " << n_cells << " " << actual << "\n";
        int written = 0;
        for (int g = 1; g <= n_genes && written < actual; ++g) {
            for (int c = 1; c <= n_cells && written < actual; ++c, ++written) {
                f << g << " " << c << " " << (g + c) << "\n";
            }
        }
    }

    // features.tsv
    {
        std::ofstream f(output_dir + "/features.tsv");
        for (int i = 1; i <= n_genes; ++i)
            f << "ENSG" << std::to_string(i) << "\tGene" << i << "\tGene Expression\n";
    }

    // barcodes.tsv
    {
        std::ofstream f(output_dir + "/barcodes.tsv");
        for (int i = 1; i <= n_cells; ++i)
            f << "ACGTACGTACGT" << std::to_string(i) << "-1\n";
    }

    return output_dir;
}

// Generate synthetic fragments.tsv.
inline std::string generate_test_fragments(const std::string& output_path,
                                           int n_fragments) {
    std::ofstream f(output_path);
    f << "# fragments file generated by singlet_pileup::interop\n";
    for (int i = 0; i < n_fragments; ++i) {
        int start = i * 200;
        int end = start + 150;
        f << "chr1\t" << start << "\t" << end << "\tACGTACGT-1\t1\n";
    }
    return output_path;
}

// Generate synthetic .pairs file.
inline std::string generate_test_pairs(const std::string& output_path,
                                       int n_contacts) {
    std::ofstream f(output_path);
    f << "## pairs format v1.0\n";
    f << "#sorted: chr1-chr2-pos1-pos2\n";
    f << "#chromosomes: chr1 chr2\n";
    f << "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n";
    for (int i = 0; i < n_contacts; ++i) {
        f << "read" << i << "\tchr1\t" << (i * 1000 + 100)
          << "\tchr1\t" << (i * 1000 + 5000) << "\t+\t-\n";
    }
    return output_path;
}

}  // namespace singlet_pileup::interop
