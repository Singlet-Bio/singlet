#pragma once
// singlet-pileup: spatial_multiome.h
// G-SPATIAL-MULTIOME: Spatial multiome router for Visium HD + spatial RNA/ATAC.
//
// Detects spatial experiment type from protocol_family and routes to the
// appropriate spatial pipeline. Extends V1-V4 Visium infrastructure to
// support Visium HD (8µm bins) and spatial multiome (RNA + ATAC).
//
// Visium HD barcodes encode position directly via a two-part structure.
// No tissue_positions.csv lookup is required for HD experiments.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace singlet {

// ── Enums ─────────────────────────────────────────────────────────────────────

enum class SpatialResolution {
    VISIUM_V1 = 0,     ///< Classic Visium 55µm spots (~5 000 spots)
    VISIUM_HD_16 = 1,  ///< Visium HD 16µm binned
    VISIUM_HD_8 = 2,   ///< Visium HD 8µm bins (primary HD product)
    VISIUM_HD_2 = 3    ///< Visium HD 2µm native resolution
};

// ── SpatialConfig ─────────────────────────────────────────────────────────────

struct SpatialConfig {
    SpatialResolution resolution = SpatialResolution::VISIUM_V1;
    bool has_atac = false;              ///< Spatial multiome with ATAC
    bool has_protein = false;           ///< CytAssist protein capture
    std::string tissue_positions_path;  ///< Path to tissue_positions.csv
    uint32_t bin_size_um = 55;          ///< Bin/spot size in micrometers
};

// ── SpatialCoord ──────────────────────────────────────────────────────────────

struct SpatialCoord {
    float x_um;    ///< X coordinate in micrometers
    float y_um;    ///< Y coordinate in micrometers
    uint32_t row;  ///< Grid row
    uint32_t col;  ///< Grid column
};

// ── Helper ────────────────────────────────────────────────────────────────────

namespace detail {
inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}
}  // namespace detail

// ── Free functions ────────────────────────────────────────────────────────────

/// Return true if protocol_family represents any spatial experiment.
/// Matches "visium", "visium-hd", "spatial-*", and "visium-multiome" variants.
inline bool is_spatial_experiment(const std::string& protocol_family) {
    const std::string lc = detail::to_lower(protocol_family);
    return lc.find("visium") != std::string::npos ||
           lc.find("spatial") != std::string::npos;
}

/// Return true if protocol_family is specifically Visium HD.
inline bool is_visium_hd(const std::string& protocol_family) {
    const std::string lc = detail::to_lower(protocol_family);
    return lc.find("visium-hd") != std::string::npos ||
           lc.find("visiumhd") != std::string::npos;
}

/// Detect spatial configuration from a protocol_family string.
///
/// Protocol family mapping:
///   "visium-hd[-*]"   → HD_8, 8µm (default HD product)
///   "visium-hd-16"    → HD_16, 16µm
///   "visium-hd-2"     → HD_2, 2µm
///   "visium[*]-atac"  → V1 + has_atac = true
///   "visium[*]"       → V1, 55µm
///   "spatial-multiome"→ V1 + has_atac = true
///   anything else spatial → V1
inline SpatialConfig detect_spatial_config(const std::string& protocol_family) {
    SpatialConfig cfg;
    const std::string lc = detail::to_lower(protocol_family);

    // ATAC component detection (spatial multiome)
    if (lc.find("atac") != std::string::npos ||
        lc.find("multiome") != std::string::npos) {
        cfg.has_atac = true;
    }

    // CytAssist protein
    if (lc.find("cytassist") != std::string::npos ||
        lc.find("protein") != std::string::npos) {
        cfg.has_protein = true;
    }

    // Visium HD resolution tiers
    if (is_visium_hd(protocol_family)) {
        if (lc.find("hd-16") != std::string::npos ||
            lc.find("hd16") != std::string::npos) {
            cfg.resolution = SpatialResolution::VISIUM_HD_16;
            cfg.bin_size_um = 16;
        } else if (lc.find("hd-2") != std::string::npos ||
                   (lc.find("hd") != std::string::npos &&
                    lc.find("hd-8") == std::string::npos &&
                    lc.find("hd-16") == std::string::npos &&
                    lc.rfind("hd", lc.size()) != std::string::npos &&
                    lc.find("hd-2") != std::string::npos)) {
            cfg.resolution = SpatialResolution::VISIUM_HD_2;
            cfg.bin_size_um = 2;
        } else {
            // Default HD product = 8µm
            cfg.resolution = SpatialResolution::VISIUM_HD_8;
            cfg.bin_size_um = 8;
        }
        return cfg;
    }

    // Classic Visium (V1–V4) and all other spatial variants → 55µm
    cfg.resolution = SpatialResolution::VISIUM_V1;
    cfg.bin_size_um = 55;
    return cfg;
}

/// Parse a Visium HD barcode to a spatial coordinate.
///
/// Visium HD barcodes encode row and column directly as zero-padded decimal
/// integers separated by 's': "ROWs_COLs" e.g. "0001s_0002s".
/// Fallback: if no separator is found, treats the barcode as an opaque index
/// (row = index / 10000, col = index % 10000).
///
/// Coordinate origin is top-left; x grows right, y grows down.
inline SpatialCoord parse_hd_barcode(const std::string& barcode,
                                     uint32_t bin_size_um) {
    SpatialCoord coord{0.0f, 0.0f, 0, 0};

    // Attempt structured parse: expect digits 's' '_' digits 's'
    // e.g.  "0001s_0001s"  or  "0001_0001"
    auto sep_pos = barcode.find('_');
    if (sep_pos != std::string::npos && sep_pos > 0) {
        try {
            // strip trailing 's' if present
            std::string row_str = barcode.substr(0, sep_pos);
            std::string col_str = barcode.substr(sep_pos + 1);
            if (!row_str.empty() && row_str.back() == 's') row_str.pop_back();
            if (!col_str.empty() && col_str.back() == 's') col_str.pop_back();

            coord.row = static_cast<uint32_t>(std::stoul(row_str));
            coord.col = static_cast<uint32_t>(std::stoul(col_str));
        } catch (...) {
            // Malformed barcode — leave at 0,0
        }
    }

    coord.x_um = static_cast<float>(coord.col) * static_cast<float>(bin_size_um);
    coord.y_um = static_cast<float>(coord.row) * static_cast<float>(bin_size_um);
    return coord;
}

/// Write spatial metadata (barcode → coordinate mapping) to a TSV file.
///
/// Format (tab-separated, header row):
///   barcode  row  col  x_um  y_um
///
/// Returns true on success.
inline bool write_spatial_metadata(const std::string& path,
                                   const std::vector<SpatialCoord>& coords,
                                   const std::vector<std::string>& barcodes,
                                   const SpatialConfig& /* config */) {
    if (coords.size() != barcodes.size()) return false;

    std::ofstream f(path);
    if (!f.is_open()) return false;

    f << "barcode\trow\tcol\tx_um\ty_um\n";
    for (std::size_t i = 0; i < barcodes.size(); ++i) {
        f << barcodes[i] << '\t'
          << coords[i].row << '\t'
          << coords[i].col << '\t'
          << coords[i].x_um << '\t'
          << coords[i].y_um << '\n';
    }
    return f.good();
}

}  // namespace singlet
