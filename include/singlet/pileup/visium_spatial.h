#pragma once
// singlet-pileup: visium_spatial.h
// V1: Visium spatial coordinate parser.
//
// Loads tissue_positions.csv produced by Space Ranger and provides barcode →
// (array_row, array_col, pixel_row, pixel_col) lookup, plus filtered exports.
//
// Two CSV formats are accepted:
//   Legacy (Space Ranger ≤1.x):
//     barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
//   Modern (Space Ranger ≥2.0 with header):
//     barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
//     (identical columns; header line auto-detected)
//
// Thread-safety: after load_positions() the object is read-only; multiple
// threads may call get_coordinate() simultaneously without synchronisation.

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

// ── SpotCoordinate ────────────────────────────────────────────────────────────

struct SpotCoordinate {
    int32_t array_row;  ///< Spot row in Visium array grid (0-based)
    int32_t array_col;  ///< Spot column in Visium array grid (0-based)
    float pixel_row;    ///< Pixel row in full-resolution image
    float pixel_col;    ///< Pixel column in full-resolution image
    bool in_tissue;     ///< true when spot is under tissue
};

// ── VisiumSpatialParser ───────────────────────────────────────────────────────

class VisiumSpatialParser {
   public:
    VisiumSpatialParser() = default;

    // ── Loading ───────────────────────────────────────────────────────────

    /// Load coordinates from a Space Ranger tissue_positions.csv file.
    /// Both headerless (Space Ranger ≤1.x) and header formats are supported.
    /// Returns true on success, false if the file could not be opened or
    /// contained no parseable coordinate entries.
    bool load_positions(const std::string& path) {
        coords_.clear();
        in_tissue_count_ = 0;

        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string line;
        bool first_line = true;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            // Detect and skip header line
            if (first_line) {
                first_line = false;
                std::string lo = line;
                for (auto& c : lo)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (lo.find("barcode") != std::string::npos ||
                    lo.find("array_row") != std::string::npos) {
                    continue;  // skip header
                }
            }

            // Parse: barcode,in_tissue,array_row,array_col,pxl_row,pxl_col
            std::vector<std::string> fields;
            fields.reserve(6);
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ','))
                fields.push_back(tok);

            if (fields.size() < 6) continue;  // malformed row

            SpotCoordinate coord;
            coord.in_tissue = (fields[1] == "1");
            coord.array_row = std::stoi(fields[2]);
            coord.array_col = std::stoi(fields[3]);
            coord.pixel_row = std::stof(fields[4]);
            coord.pixel_col = std::stof(fields[5]);

            const std::string& bc = fields[0];
            if (coord.in_tissue) ++in_tissue_count_;
            coords_[bc] = coord;
        }
        return !coords_.empty();
    }

    // ── Lookup ────────────────────────────────────────────────────────────

    /// Get the coordinate for a barcode, or nullptr if not found.
    const SpotCoordinate* get_coordinate(const std::string& barcode) const {
        auto it = coords_.find(barcode);
        return (it != coords_.end()) ? &it->second : nullptr;
    }

    // ── Export ────────────────────────────────────────────────────────────

    /// Write a spatial metadata TSV with columns:
    ///   barcode  array_row  array_col  pixel_row  pixel_col  in_tissue
    /// Barcodes not found in the positions file receive all-zero coordinates.
    void write_coordinates_tsv(const std::string& path,
                               const std::vector<std::string>& barcodes) const {
        std::ofstream ofs(path);
        ofs << "barcode\tarray_row\tarray_col\tpixel_row\tpixel_col\tin_tissue\n";
        for (const auto& bc : barcodes) {
            const SpotCoordinate* c = get_coordinate(bc);
            if (c) {
                ofs << bc << '\t'
                    << c->array_row << '\t'
                    << c->array_col << '\t'
                    << c->pixel_row << '\t'
                    << c->pixel_col << '\t'
                    << (c->in_tissue ? 1 : 0) << '\n';
            } else {
                ofs << bc << "\t0\t0\t0\t0\t0\n";
            }
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────

    /// Total number of spots loaded (tissue + non-tissue).
    size_t total_spots() const { return coords_.size(); }

    /// Number of spots with in_tissue == true.
    size_t in_tissue() const { return in_tissue_count_; }

   private:
    std::unordered_map<std::string, SpotCoordinate> coords_;
    size_t in_tissue_count_ = 0;
};

}  // namespace singlet
