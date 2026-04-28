#pragma once
// singlet-pileup: spatial_barcodes.h
// G-SPATIAL-BARCODES: Spatial transcriptomics barcode decoders.
//
// Supports Slide-seq (14bp), Stereo-seq (25bp DNB CID), Visium HD, HDST, MERFISH.
// Factory pattern: make_decoder(SpatialBarcodeFormat) → SpatialBarcodeDecoder.
//
// Thread-safety: decoders are stateless after construction; safe to call
// decode() from multiple threads simultaneously.

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace singlet {

// ── Types ─────────────────────────────────────────────────────────────────────

enum class SpatialBarcodeFormat {
    SLIDESEQ,   ///< Slide-seq v1/v2 — 14 bp barcode → bead location lookup
    STEREOSEQ,  ///< Stereo-seq DNB CID — 25 bp, x/y encoded in first 25 bits
    VISIUM_HD,  ///< 10x Visium HD — 16 bp BC, coordinate from barcode file
    HDST,       ///< HDST — 20 bp barcode
    MERFISH,    ///< MERFISH bit-plane encoded barcode (categorical)
    UNKNOWN
};

struct SpatialCoord {
    double x;
    double y;
    uint32_t section_id = 0;

    bool valid = true;  ///< false for undecodable / empty barcodes

    bool operator==(const SpatialCoord& o) const {
        return x == o.x && y == o.y && section_id == o.section_id && valid == o.valid;
    }
};

// ── base decoder ─────────────────────────────────────────────────────────────

class SpatialBarcodeDecoder {
   public:
    virtual ~SpatialBarcodeDecoder() = default;
    virtual SpatialCoord decode(std::string_view bc) const = 0;
    virtual SpatialBarcodeFormat format() const = 0;
};

// ── Slide-seq decoder ─────────────────────────────────────────────────────────
// Slide-seq uses 14 bp BeadBarcodes whose true (x,y) coordinates come from a
// bead-location file.  When no file is available we extract a hash-based
// placeholder coordinate from the barcode bits so that the data model is still
// exercisable end-to-end.

class SlideseqDecoder : public SpatialBarcodeDecoder {
   public:
    SpatialBarcodeFormat format() const override { return SpatialBarcodeFormat::SLIDESEQ; }

    SpatialCoord decode(std::string_view bc) const override {
        if (bc.empty()) return {0.0, 0.0, 0, false};
        // Encode each base: A→0, C→1, G→2, T→3
        uint32_t bits = 0;
        for (char c : bc) {
            bits = bits * 4 + base_to_bits(c);
        }
        // Distribute 28 low bits evenly between x and y
        double x = static_cast<double>(bits & 0x3FFF);          // low 14 bits
        double y = static_cast<double>((bits >> 14) & 0x3FFF);  // next 14 bits
        return {x, y, 0, true};
    }

   private:
    static uint32_t base_to_bits(char c) {
        switch (c) {
            case 'A':
            case 'a':
                return 0;
            case 'C':
            case 'c':
                return 1;
            case 'G':
            case 'g':
                return 2;
            case 'T':
            case 't':
                return 3;
            default:
                return 0;
        }
    }
};

// ── Stereo-seq decoder ────────────────────────────────────────────────────────
// Stereo-seq DNB Coordinate ID (CID): 25 bp barcode encodes x in bits [24:12]
// and y in bits [11:0] after converting DNA→binary (A/C=0, G/T=1 per DNB spec).

class StereoseqDecoder : public SpatialBarcodeDecoder {
   public:
    SpatialBarcodeFormat format() const override { return SpatialBarcodeFormat::STEREOSEQ; }

    SpatialCoord decode(std::string_view bc) const override {
        if (bc.empty()) return {0.0, 0.0, 0, false};
        if (bc.size() > 25) bc = bc.substr(0, 25);

        uint32_t bits = 0;
        for (size_t i = 0; i < bc.size(); ++i) {
            // DNB encoding: A/C → 0, G/T → 1
            uint32_t b = (bc[i] == 'G' || bc[i] == 'g' ||
                          bc[i] == 'T' || bc[i] == 't')
                             ? 1u
                             : 0u;
            bits = (bits << 1) | b;
        }
        // Pad up to 25 bits if input was shorter
        uint32_t shift = static_cast<uint32_t>(25 - bc.size());
        bits <<= shift;

        uint32_t x = (bits >> 12) & 0x1FFF;  // bits [24:12] → 13 bits
        uint32_t y = bits & 0x0FFF;          // bits [11:0]  → 12 bits
        return {static_cast<double>(x), static_cast<double>(y), 0, true};
    }
};

// ── Visium HD decoder ─────────────────────────────────────────────────────────
// Visium HD 16 bp barcodes encode a 2 µm bin coordinate; without the barcode
// allowlist we use the same hash-based approach as Slide-seq.

class VisiumHdDecoder : public SpatialBarcodeDecoder {
   public:
    SpatialBarcodeFormat format() const override { return SpatialBarcodeFormat::VISIUM_HD; }

    SpatialCoord decode(std::string_view bc) const override {
        if (bc.empty()) return {0.0, 0.0, 0, false};
        uint32_t h = 2166136261u;
        for (char c : bc) h = (h ^ static_cast<uint8_t>(c)) * 16777619u;
        double x = static_cast<double>(h & 0xFFFF);
        double y = static_cast<double>((h >> 16) & 0xFFFF);
        return {x, y, 0, true};
    }
};

// ── HDST decoder ─────────────────────────────────────────────────────────────
// HDST 20 bp barcodes — hash-based placeholder.

class HdstDecoder : public SpatialBarcodeDecoder {
   public:
    SpatialBarcodeFormat format() const override { return SpatialBarcodeFormat::HDST; }

    SpatialCoord decode(std::string_view bc) const override {
        if (bc.empty()) return {0.0, 0.0, 0, false};
        uint32_t h = 0;
        for (char c : bc) h = h * 31 + static_cast<uint8_t>(c);
        double x = static_cast<double>(h & 0x3FF);
        double y = static_cast<double>((h >> 10) & 0x3FF);
        return {x, y, 0, true};
    }
};

// ── MERFISH decoder ───────────────────────────────────────────────────────────
// MERFISH encodes gene identity in bit-plane sequences; this decoder maps the
// barcode string to a categorical index (x = gene index, y = 0).

class MerfishDecoder : public SpatialBarcodeDecoder {
   public:
    SpatialBarcodeFormat format() const override { return SpatialBarcodeFormat::MERFISH; }

    SpatialCoord decode(std::string_view bc) const override {
        if (bc.empty()) return {0.0, 0.0, 0, false};
        uint32_t idx = 0;
        for (char c : bc) idx = idx * 2 + (c == '1' ? 1u : 0u);
        return {static_cast<double>(idx), 0.0, 0, true};
    }
};

// ── Factory ───────────────────────────────────────────────────────────────────

/// Create a heap-allocated decoder for the given format.
inline std::unique_ptr<SpatialBarcodeDecoder> make_decoder(SpatialBarcodeFormat fmt) {
    switch (fmt) {
        case SpatialBarcodeFormat::SLIDESEQ:
            return std::make_unique<SlideseqDecoder>();
        case SpatialBarcodeFormat::STEREOSEQ:
            return std::make_unique<StereoseqDecoder>();
        case SpatialBarcodeFormat::VISIUM_HD:
            return std::make_unique<VisiumHdDecoder>();
        case SpatialBarcodeFormat::HDST:
            return std::make_unique<HdstDecoder>();
        case SpatialBarcodeFormat::MERFISH:
            return std::make_unique<MerfishDecoder>();
        default:
            throw std::invalid_argument("make_decoder: unknown SpatialBarcodeFormat");
    }
}

// ── Convenience free functions ────────────────────────────────────────────────

/// Hash-based placeholder decoder for Slide-seq without a bead-location file.
inline SpatialCoord decode_slideseq_barcode(std::string_view bc) {
    return SlideseqDecoder{}.decode(bc);
}

/// DNB CID decoder for Stereo-seq 25 bp barcodes.
inline SpatialCoord decode_stereoseq_barcode(std::string_view bc) {
    return StereoseqDecoder{}.decode(bc);
}

// ── Auto-detection ────────────────────────────────────────────────────────────

/// Infer spatial platform from barcode length, metadata availability, and
/// any protocol tag already parsed by the detection pipeline.
inline SpatialBarcodeFormat detect_spatial_platform(int cblen,
                                                    bool has_spatial_file,
                                                    std::string_view protocol_tag) {
    // Explicit protocol tag takes highest priority.
    if (protocol_tag == "stereoseq" || protocol_tag == "stereo-seq") return SpatialBarcodeFormat::STEREOSEQ;
    if (protocol_tag == "slideseq" || protocol_tag == "slide-seq") return SpatialBarcodeFormat::SLIDESEQ;
    if (protocol_tag == "visium-hd" || protocol_tag == "visiumhd") return SpatialBarcodeFormat::VISIUM_HD;
    if (protocol_tag == "hdst") return SpatialBarcodeFormat::HDST;
    if (protocol_tag == "merfish") return SpatialBarcodeFormat::MERFISH;

    // Barcode length disambiguation.
    if (cblen == 25) return SpatialBarcodeFormat::STEREOSEQ;
    if (cblen == 20) return SpatialBarcodeFormat::HDST;
    if (cblen == 16 && has_spatial_file) return SpatialBarcodeFormat::VISIUM_HD;
    if (cblen == 14 && has_spatial_file) return SpatialBarcodeFormat::SLIDESEQ;

    return SpatialBarcodeFormat::UNKNOWN;
}

}  // namespace singlet
