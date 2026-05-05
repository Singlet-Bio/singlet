// test_multiome_router.cpp
// Unit tests for singlet::multiome_router (G-MULTIOME-ROUTE)
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "singlet-pileup/multiome_router.h"

using namespace singlet;

// ── detect_modality ──────────────────────────────────────────────────────────

static void test_detect_arc_gex() {
    assert(detect_modality("10x-arc-gex") == MultiomeModality::GEX);
}

static void test_detect_arc_atac() {
    assert(detect_modality("10x-arc-atac") == MultiomeModality::ATAC);
}

static void test_detect_multiome_gex() {
    assert(detect_modality("10x-multiome-gex") == MultiomeModality::GEX);
}

static void test_detect_multiome_atac() {
    assert(detect_modality("10x-multiome-atac") == MultiomeModality::ATAC);
}

// Non-multiome GEX protocol → still GEX modality
static void test_detect_10xv3_is_gex() {
    assert(detect_modality("10x-3p-v3") == MultiomeModality::GEX);
}

static void test_detect_dropseq_is_gex() {
    assert(detect_modality("dropseq") == MultiomeModality::GEX);
}

// Generic atac substring detection
static void test_detect_atac_substring() {
    assert(detect_modality("10x-atac") == MultiomeModality::ATAC);
    assert(detect_modality("custom-atac-v2") == MultiomeModality::ATAC);
}

static void test_detect_empty_is_unknown() {
    assert(detect_modality("") == MultiomeModality::UNKNOWN);
}

// ── is_multiome_protocol ─────────────────────────────────────────────────────

static void test_is_multiome_arc_gex() {
    assert(is_multiome_protocol("10x-arc-gex") == true);
}

static void test_is_multiome_arc_atac() {
    assert(is_multiome_protocol("10x-arc-atac") == true);
}

static void test_is_multiome_10xv3_false() {
    assert(is_multiome_protocol("10x-3p-v3") == false);
}

static void test_is_multiome_dropseq_false() {
    assert(is_multiome_protocol("dropseq") == false);
}

static void test_is_multiome_multiome_alias() {
    assert(is_multiome_protocol("multiome") == true);
    assert(is_multiome_protocol("10xmultiome") == true);
    assert(is_multiome_protocol("10xarcmultiome") == true);
}

// ── classify_inputs (no real .1fq files needed — pass empty list) ────────────

static void test_classify_empty() {
    MultiomeInputs r = classify_inputs({});
    assert(r.gex_inputs.empty());
    assert(r.atac_inputs.empty());
    assert(r.is_multiome == false);
}

// Simulate mixed GEX+ATAC by writing tiny fake .1fq files whose byte[57]
// encodes the appropriate AssayType.
static std::string write_fake_1fq(const char* name, lib1fq::AssayType at) {
    std::string path = std::string("/dev/shm/") + name;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return "";
    // Write 64 zero bytes and set byte 57 to the assay type
    uint8_t buf[64] = {};
    // Set magic bytes (first 8 bytes) to something non-zero so fread works
    buf[0] = 0x1f;
    buf[1] = 0x71;  // fake magic — enough for our peek logic
    buf[57] = static_cast<uint8_t>(at);
    std::fwrite(buf, 1, sizeof(buf), f);
    std::fclose(f);
    return path;
}

static void test_classify_gex_only() {
    std::string p = write_fake_1fq("tmr_gex.1fq", lib1fq::AssayType::SC_MULTIOME_GEX);
    if (p.empty()) return;  // skip if /dev/shm unavailable

    MultiomeInputs r = classify_inputs({p});
    assert(r.gex_inputs.size() == 1);
    assert(r.atac_inputs.empty());
    assert(r.is_multiome == false);
    std::remove(p.c_str());
}

static void test_classify_atac_only() {
    std::string p = write_fake_1fq("tmr_atac.1fq", lib1fq::AssayType::SC_MULTIOME_ATAC);
    if (p.empty()) return;

    MultiomeInputs r = classify_inputs({p});
    assert(r.gex_inputs.empty());
    assert(r.atac_inputs.size() == 1);
    assert(r.is_multiome == false);
    std::remove(p.c_str());
}

static void test_classify_mixed_is_multiome() {
    std::string pg = write_fake_1fq("tmr_mix_gex.1fq", lib1fq::AssayType::SC_MULTIOME_GEX);
    std::string pa = write_fake_1fq("tmr_mix_atac.1fq", lib1fq::AssayType::SC_MULTIOME_ATAC);
    if (pg.empty() || pa.empty()) return;

    MultiomeInputs r = classify_inputs({pg, pa});
    assert(r.gex_inputs.size() == 1);
    assert(r.atac_inputs.size() == 1);
    assert(r.is_multiome == true);
    std::remove(pg.c_str());
    std::remove(pa.c_str());
}

// ── plan_multiome_run ────────────────────────────────────────────────────────

static void test_plan_single_gex() {
    MultiomeInputs inp;
    inp.gex_inputs = {"a.1fq"};
    inp.is_multiome = false;

    MultiomeConfig cfg = plan_multiome_run(inp, "/out");
    assert(cfg.run_gex == true);
    assert(cfg.run_atac == false);
    // Single modality: prefix used as-is (with trailing /)
    assert(cfg.gex_output_prefix == "/out/");
}

static void test_plan_single_atac() {
    MultiomeInputs inp;
    inp.atac_inputs = {"b.1fq"};
    inp.is_multiome = false;

    MultiomeConfig cfg = plan_multiome_run(inp, "/out");
    assert(cfg.run_gex == false);
    assert(cfg.run_atac == true);
    assert(cfg.atac_output_prefix == "/out/");
}

static void test_plan_multiome_uses_subdirs() {
    MultiomeInputs inp;
    inp.gex_inputs = {"a.1fq"};
    inp.atac_inputs = {"b.1fq"};
    inp.is_multiome = true;

    MultiomeConfig cfg = plan_multiome_run(inp, "/out/sample");
    assert(cfg.run_gex == true);
    assert(cfg.run_atac == true);
    assert(cfg.gex_output_prefix == "/out/sample/gex/");
    assert(cfg.atac_output_prefix == "/out/sample/atac/");
}

static void test_plan_trailing_slash_idempotent() {
    MultiomeInputs inp;
    inp.gex_inputs = {"a.1fq"};
    inp.is_multiome = false;

    MultiomeConfig cfg = plan_multiome_run(inp, "/out/");
    assert(cfg.gex_output_prefix == "/out/");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_detect_arc_gex();
    test_detect_arc_atac();
    test_detect_multiome_gex();
    test_detect_multiome_atac();
    test_detect_10xv3_is_gex();
    test_detect_dropseq_is_gex();
    test_detect_atac_substring();
    test_detect_empty_is_unknown();

    test_is_multiome_arc_gex();
    test_is_multiome_arc_atac();
    test_is_multiome_10xv3_false();
    test_is_multiome_dropseq_false();
    test_is_multiome_multiome_alias();

    test_classify_empty();
    test_classify_gex_only();
    test_classify_atac_only();
    test_classify_mixed_is_multiome();

    test_plan_single_gex();
    test_plan_single_atac();
    test_plan_multiome_uses_subdirs();
    test_plan_trailing_slash_idempotent();

    std::puts("All multiome_router tests passed.");
    return 0;
}
