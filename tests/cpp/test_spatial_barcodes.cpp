// SPDX-License-Identifier: MIT
// test/test_spatial_barcodes.cpp — Unit tests for G-SPATIAL-BARCODES
// Tests: SlideseqDecoder, StereoseqDecoder, VisiumHdDecoder, HdstDecoder,
//        MerfishDecoder, detect_spatial_platform, make_decoder factory.

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include "singlet/pileup/spatial_barcodes.h"

using namespace singlet;

#define SINGLET_TEST_HARNESS_QUIET
#include "test_harness.h"  // CHECK(cond, msg) + g_pass / g_fail

// ── Slide-seq ─────────────────────────────────────────────────────────────────

static void test_slideseq_basic() {
    auto coord = decode_slideseq_barcode("AAAAAAAAAAAAA");  // 13 A's
    CHECK(coord.valid, "slideseq: 13-base valid");
    CHECK(coord.x >= 0.0, "slideseq: x non-negative");
    CHECK(coord.y >= 0.0, "slideseq: y non-negative");
}

static void test_slideseq_14bp() {
    auto coord = decode_slideseq_barcode("ACGTACGTACGTAC");  // 14 bp
    CHECK(coord.valid, "slideseq: 14bp decode valid");
    CHECK(coord.x != coord.y || coord.x == 0.0, "slideseq: x/y differ (or both 0)");
}

static void test_slideseq_empty() {
    auto coord = decode_slideseq_barcode("");
    CHECK(!coord.valid, "slideseq: empty barcode invalid");
    CHECK(coord.x == 0.0 && coord.y == 0.0, "slideseq: empty barcode zero coords");
}

static void test_slideseq_deterministic() {
    auto a = decode_slideseq_barcode("ACGTACGTACGTAC");
    auto b = decode_slideseq_barcode("ACGTACGTACGTAC");
    CHECK(a == b, "slideseq: deterministic output");
}

// ── Stereo-seq ────────────────────────────────────────────────────────────────

static void test_stereoseq_25bp() {
    // 25 G/T → all bits 1 → bits = 0x1FFFFFF
    // x = bits[24:12] = 0x1FFF = 8191, y = bits[11:0] = 0xFFF = 4095
    auto coord = decode_stereoseq_barcode("GGGGGGGGGGGGGGGGGGGGGGGGG");
    CHECK(coord.valid, "stereoseq: 25 G valid");
    CHECK(coord.x == 8191.0, "stereoseq: max x = 8191");
    CHECK(coord.y == 4095.0, "stereoseq: max y = 4095");
}

static void test_stereoseq_all_ac() {
    // 25 A/C → all bits 0 → x=0, y=0
    auto coord = decode_stereoseq_barcode("AAAAAAAAAAAAAAAAAAAAAAAAA");
    CHECK(coord.valid, "stereoseq: 25 A valid");
    CHECK(coord.x == 0.0, "stereoseq: zero x");
    CHECK(coord.y == 0.0, "stereoseq: zero y");
}

static void test_stereoseq_empty() {
    auto coord = decode_stereoseq_barcode("");
    CHECK(!coord.valid, "stereoseq: empty invalid");
}

static void test_stereoseq_shorter_input() {
    // Shorter than 25 — should still produce a valid (padded) result
    auto coord = decode_stereoseq_barcode("GTGTGTGTGT");
    CHECK(coord.valid, "stereoseq: short input valid");
}

static void test_stereoseq_deterministic() {
    auto a = decode_stereoseq_barcode("ATCGATCGATCGATCGATCGATCGA");
    auto b = decode_stereoseq_barcode("ATCGATCGATCGATCGATCGATCGA");
    CHECK(a == b, "stereoseq: deterministic");
}

// ── Visium HD ─────────────────────────────────────────────────────────────────

static void test_visiumhd_basic() {
    auto dec = make_decoder(SpatialBarcodeFormat::VISIUM_HD);
    auto coord = dec->decode("ACGTACGTACGTACGT");  // 16 bp
    CHECK(coord.valid, "visiumhd: 16bp valid");
    CHECK(coord.x >= 0.0, "visiumhd: x >= 0");
}

static void test_visiumhd_empty() {
    VisiumHdDecoder dec;
    auto coord = dec.decode("");
    CHECK(!coord.valid, "visiumhd: empty invalid");
}

// ── HDST ──────────────────────────────────────────────────────────────────────

static void test_hdst_20bp() {
    auto dec = make_decoder(SpatialBarcodeFormat::HDST);
    auto coord = dec->decode("ACGTACGTACGTACGTACGT");  // 20 bp
    CHECK(coord.valid, "hdst: 20bp valid");
    CHECK(dec->format() == SpatialBarcodeFormat::HDST, "hdst: format tag");
}

static void test_hdst_empty() {
    HdstDecoder dec;
    CHECK(!dec.decode("").valid, "hdst: empty invalid");
}

// ── MERFISH ───────────────────────────────────────────────────────────────────

static void test_merfish_bits() {
    // "101" in binary == 5; x should be 5, y = 0
    MerfishDecoder dec;
    auto coord = dec.decode("101");
    CHECK(coord.valid, "merfish: bit string valid");
    CHECK(coord.x == 5.0, "merfish: bit-encoded x");
    CHECK(coord.y == 0.0, "merfish: y always 0");
}

static void test_merfish_all_zero() {
    MerfishDecoder dec;
    auto coord = dec.decode("000");
    CHECK(coord.x == 0.0, "merfish: all-zero x = 0");
}

// ── detect_spatial_platform ───────────────────────────────────────────────────

static void test_detect_by_tag() {
    CHECK(detect_spatial_platform(0, false, "stereoseq") == SpatialBarcodeFormat::STEREOSEQ,
          "detect: stereoseq tag");
    CHECK(detect_spatial_platform(0, false, "slideseq") == SpatialBarcodeFormat::SLIDESEQ,
          "detect: slideseq tag");
    CHECK(detect_spatial_platform(0, false, "visium-hd") == SpatialBarcodeFormat::VISIUM_HD,
          "detect: visium-hd tag");
    CHECK(detect_spatial_platform(0, false, "hdst") == SpatialBarcodeFormat::HDST,
          "detect: hdst tag");
    CHECK(detect_spatial_platform(0, false, "merfish") == SpatialBarcodeFormat::MERFISH,
          "detect: merfish tag");
}

static void test_detect_by_cblen() {
    CHECK(detect_spatial_platform(25, false, "") == SpatialBarcodeFormat::STEREOSEQ,
          "detect: cblen=25 → stereoseq");
    CHECK(detect_spatial_platform(20, false, "") == SpatialBarcodeFormat::HDST,
          "detect: cblen=20 → hdst");
    CHECK(detect_spatial_platform(16, true, "") == SpatialBarcodeFormat::VISIUM_HD,
          "detect: cblen=16+file → visium_hd");
    CHECK(detect_spatial_platform(14, true, "") == SpatialBarcodeFormat::SLIDESEQ,
          "detect: cblen=14+file → slideseq");
    CHECK(detect_spatial_platform(8, false, "") == SpatialBarcodeFormat::UNKNOWN,
          "detect: unknown len → UNKNOWN");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_slideseq_basic();
    test_slideseq_14bp();
    test_slideseq_empty();
    test_slideseq_deterministic();

    test_stereoseq_25bp();
    test_stereoseq_all_ac();
    test_stereoseq_empty();
    test_stereoseq_shorter_input();
    test_stereoseq_deterministic();

    test_visiumhd_basic();
    test_visiumhd_empty();

    test_hdst_20bp();
    test_hdst_empty();

    test_merfish_bits();
    test_merfish_all_zero();

    test_detect_by_tag();
    test_detect_by_cblen();

    std::cout << "spatial_barcodes: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail > 0) ? 1 : 0;
}
