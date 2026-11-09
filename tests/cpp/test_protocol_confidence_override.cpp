// SPDX-License-Identifier: MIT
// test_protocol_confidence_override.cpp
// The detector-vs-GEO policy (protocol.h resolve_protocol_call), the GEM-X lists and the 28-nt fallback of detect_protocol.
//
// The campaign passes each sample's GEO chemistry as "protocol" in --metadata-json (EncoderConfig::metadata_protocol):
//   1. confident detection (MEDIUM/HIGH) stands when GEO disagrees;
//   2. low-confidence detection (LOW/NONE) that disagrees with GEO: both calls are tested on the same probe reads (whitelist hit rate at
//      or above kProtocolWlRateFloor, plus the 3'/5' strand check for a 10x pair that shares one list); the one that passes is taken,
//      and when both pass or neither does the run is excluded with the reason recorded;
//   3. the decision and its evidence are written to the metadata JSON (protocol_provenance_json);
//   4. GEM-X (10x-3p-v4, 10x-5p-v4) reads its own fetched lists; without them a GEM-X call fails and names `singlet whitelists fetch`.
// Whitelists here are small synthetic lists written to a temporary directory under the real file names.

#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#include "singlet/fq/protocol.h"
#include "singlet/fq/types.h"
#include "test_harness.h"

using namespace singlet::fq;

namespace {

struct Spot {
    std::vector<uint8_t> r1_seq, r2_seq;
    uint16_t r1_len = 0, r2_len = 0;
};

const char* kNuc = "ACGT";

std::vector<std::string> make_list(uint64_t seed, size_t n) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> v;
    for (size_t i = 0; i < n; ++i) {
        std::string s(16, 'A');
        for (auto& c : s) c = kNuc[rng() & 3];
        v.push_back(s);
    }
    return v;
}

void write_list(const std::string& path, const std::vector<std::string>& v) {
    std::ofstream o(path);
    for (const auto& s : v) o << s << "\n";
}

std::vector<uint8_t> codes(const std::string& s) {
    std::vector<uint8_t> v;
    for (char c : s) v.push_back(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : 3);
    return v;
}

// 10x-like reads: R1 = barcode (from `list`, or random when empty) + UMI(umi_len) [+ 13-nt tail], R2 = 90-nt cDNA.
// strand "3p": 3'-kit TSO in 2% of R2 reads and poly(dT) after the UMI; "5p": the 5' TSO tail after the UMI; "": no tail.
std::vector<Spot> make_spots(const std::vector<std::string>& list, uint16_t umi_len, const std::string& strand, uint64_t seed,
                             size_t n = 2000) {
    std::mt19937_64 rng(seed);
    std::vector<Spot> out(n);
    for (size_t i = 0; i < n; ++i) {
        std::string r1 = list.empty() ? std::string() : list[rng() % list.size()];
        while (r1.size() < 16) r1 += kNuc[rng() & 3];
        for (uint16_t k = 0; k < umi_len; ++k) r1 += kNuc[rng() & 3];
        if (strand == "3p") r1 += "TTTTTTTTTTTTT";
        if (strand == "5p") r1 += "TTTCTTATATGGG";
        std::string r2;
        for (int k = 0; k < 90; ++k) r2 += kNuc[rng() & 3];
        if (strand == "3p" && i % 50 == 0) r2.replace(20, 13, "CAGAGTACATGGG");
        out[i].r1_seq = codes(r1);
        out[i].r2_seq = codes(r2);
        out[i].r1_len = static_cast<uint16_t>(r1.size());
        out[i].r2_len = static_cast<uint16_t>(r2.size());
    }
    return out;
}

ProtocolCallTest passed(const std::string& tag, double rate) { return {tag, rate, 2000, "", true, ""}; }
ProtocolCallTest failed(const std::string& tag, const std::string& why) { return {tag, 0.01, 2000, "", false, why}; }

ProtocolDecision pre(const std::string& det, Confidence c, const std::string& geo, ProtocolCallTest dt, ProtocolCallTest gt) {
    ProtocolDecision d;
    d.detector_tag = det;
    d.detector_confidence = c;
    d.geo_raw = geo;
    if (const CandidateSpec* s = geo.empty() ? nullptr : find_protocol_spec(geo)) d.geo_tag = s->tag;
    d.detector_test = dt;
    d.geo_test = gt;
    return d;
}

}  // namespace

int main() {
    // ── decide_protocol_call: every branch of the policy ──
    {
        auto d = decide_protocol_call(pre("10x-3p-v3", Confidence::HIGH, "10x-3p-v2", passed("10x-3p-v3", 0.9), passed("10x-3p-v2", 0.5)));
        CHECK(d.source == "detector" && d.tag == "10x-3p-v3", "HIGH confidence: the detector wins over a disagreeing GEO chemistry");
        d = decide_protocol_call(pre("10x-3p-v3", Confidence::MEDIUM, "10x-3p-v2",
                                     ProtocolCallTest{"10x-3p-v3", 0.8, 2000, "5p", false, "strand check: the cDNA read looks 5'"},
                                     passed("10x-3p-v2", 0.6)));
        CHECK(d.source == "detector" && d.tag == "10x-3p-v3",
              "MEDIUM confidence above the floor: the detector wins even when only GEO's call passes");
        d = decide_protocol_call(pre("10x-3p-v1", Confidence::MEDIUM, "indrop", failed("10x-3p-v1", "0.1% of barcode reads"),
                                     failed("indrop", "indrop has no barcode whitelist to test against")));
        CHECK(d.source == "excluded", "MEDIUM call below its own whitelist floor counts as low confidence: tested against GEO, excluded");
        d = decide_protocol_call(pre("10x-3p-v3", Confidence::LOW, "10x_3p_v2", failed("10x-3p-v3", "2% of barcode reads"), passed("10x-3p-v2", 0.6)));
        CHECK(d.source == "geo" && d.tag == "10x-3p-v2", "LOW confidence, only GEO passes: GEO's chemistry (alias resolved)");
        d = decide_protocol_call(pre("10x-3p-v3", Confidence::LOW, "10x-3p-v2", passed("10x-3p-v3", 0.7), failed("10x-3p-v2", "3%")));
        CHECK(d.source == "detector" && d.tag == "10x-3p-v3", "LOW confidence, only the detector passes: the detector");
        d = decide_protocol_call(pre("10x-3p-v3", Confidence::LOW, "10x-3p-v4", passed("10x-3p-v3", 0.7), passed("10x-3p-v4", 0.7)));
        CHECK(d.source == "excluded" && d.tag.empty() && d.exclusion_reason.find("both pass") != std::string::npos,
              "LOW confidence, both pass and disagree: excluded, reason recorded");
        d = decide_protocol_call(pre("dropseq", Confidence::LOW, "seqwell", failed("dropseq", "dropseq has no barcode whitelist to test against"),
                                     failed("seqwell", "seqwell has no barcode whitelist to test against")));
        CHECK(d.source == "excluded" && d.exclusion_reason.find("both fail") != std::string::npos &&
              d.exclusion_reason.find("dropseq has no barcode whitelist") != std::string::npos,
              "LOW confidence, neither passes: excluded, with both reasons");
        d = decide_protocol_call(pre("", Confidence::NONE, "10x-3p-v2", ProtocolCallTest{}, passed("10x-3p-v2", 0.8)));
        CHECK(d.source == "geo" && d.tag == "10x-3p-v2", "no call (NONE) and GEO passes: GEO");
        d = decide_protocol_call(pre("10x-visium", Confidence::LOW, "", failed("10x-visium", "x"), ProtocolCallTest{}));
        CHECK(d.source == "detector" && d.tag == "10x-visium", "no GEO chemistry: the detector, whatever its confidence");
        d = decide_protocol_call(pre("10x-3p-v3", Confidence::LOW, "10xv3", passed("10x-3p-v3", 0.4), ProtocolCallTest{}));
        CHECK(d.source == "detector", "GEO agrees (alias 10xv3): the detector");
        d = decide_protocol_call(pre("10x-3p-v3", Confidence::LOW, "totally-made-up", passed("10x-3p-v3", 0.4), ProtocolCallTest{}));
        CHECK(d.source == "detector" && d.geo_tag.empty(), "an unknown GEO name is recorded but not tested: the detector");
    }

    // ── resolve_protocol_call on reads, with whitelists on disk ──
    char tmpl[] = "/tmp/singlet-confpol-XXXXXX";
    const std::string wl = mkdtemp(tmpl);
    const auto L3 = make_list(1, 300), L2 = make_list(2, 300), L4 = make_list(3, 300), L5 = make_list(4, 300);
    write_list(wl + "/3M-february-2018.txt", L3);
    write_list(wl + "/737K-august-2016.txt", L2);
    const std::vector<std::string> dirs{wl};
    {
        auto spots = make_spots(L2, 10, "3p", 11);   // 3' v2 reads
        auto d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x_3p_v2", spots, dirs);
        CHECK(d.source == "geo" && d.tag == "10x-3p-v2" && d.geo_test.wl_rate > 0.9 && d.detector_test.wl_rate < 0.1,
              "v2 reads, LOW v3 call, GEO v2: GEO's call passes the floor, the detector's fails");
        d = resolve_protocol_call("10x-3p-v3", Confidence::HIGH, "10x_3p_v2", spots, dirs);
        CHECK(d.source == "geo", "the same reads with a HIGH v3 call below its own floor: not confident, GEO's v2 passes");
        d = resolve_protocol_call("10x-3p-v2", Confidence::HIGH, "10x-3p-v3", spots, dirs);
        CHECK(d.source == "detector" && d.geo_test.wl_rate >= 0 && d.geo_test.wl_rate < 0.1,
              "a HIGH v2 call that passes stands against GEO v3, and GEO's whitelist rate is recorded");
    }
    {
        auto five = make_spots(L3, 12, "5p", 12);    // 5' reads on the shared 3M list
        auto d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x-5p-v3", five, dirs);
        CHECK(d.source == "geo" && d.tag == "10x-5p-v3" && d.detector_test.why.find("strand") != std::string::npos,
              "5' reads, LOW 3' call, GEO 5': the strand check fails the 3' call and passes GEO's");
        auto three = make_spots(L3, 12, "3p", 13);
        d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x-5p-v3", three, dirs);
        CHECK(d.source == "detector" && d.geo_test.why.find("strand") != std::string::npos,
              "3' reads, LOW 3' call, GEO 5': GEO's call fails the strand check, the detector's stands");
        CHECK(d.strand_measured && d.strand_evidence.r2_smart > 0, "strand evidence is kept for the provenance");
    }
    {
        auto junk = make_spots({}, 12, "", 14);      // barcodes in no list
        auto d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x-3p-v2", junk, dirs);
        CHECK(d.source == "excluded" && d.exclusion_reason.find("both fail") != std::string::npos &&
              d.exclusion_reason.find("floor 10%") != std::string::npos,
              "reads in neither list: excluded, the reason names the floor");
        const std::string j = protocol_provenance_json(d);
        CHECK(j.find("\"protocol_source\": \"excluded\"") != std::string::npos && j.find("\"protocol_confidence\": 1") != std::string::npos &&
              j.find("\"protocol_exclusion_reason\": \"low-confidence") != std::string::npos &&
              j.find("\"detector_wl_rate\": 0.0") != std::string::npos && j.find("\"geo_wl_rate\": 0.0") != std::string::npos,
              "provenance JSON: source, confidence, exclusion reason and both whitelist rates");
    }
    {   // GEM-X: lists absent
        auto v4 = make_spots(L4, 12, "3p", 15);
        auto d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x-3p-v4", v4, dirs);
        CHECK(d.source == "excluded" && d.geo_test.why.find("3M-3pgex-may-2023.txt is not installed (singlet whitelists fetch)") != std::string::npos,
              "GEO GEM-X without its list: the GEM-X call cannot pass, and the reason names the fetch command");
        const std::string e = fetched_whitelist_missing_error("10x-3p-v4", dirs);
        CHECK(e.find("3M-3pgex-may-2023.txt") != std::string::npos && e.find("singlet whitelists fetch") != std::string::npos &&
              e.find("No other list is substituted") != std::string::npos,
              "a 10x-3p-v4 call without its list fails clearly and names the fetch command");
        CHECK(fetched_whitelist_missing_error("10x-5p-v4", dirs).find("3M-5pgex-jan-2023.txt") != std::string::npos,
              "10x-5p-v4 (GEM-X 5' v3) likewise");
        CHECK(fetched_whitelist_missing_error("10x-3p-v3", dirs).empty(), "bundled chemistries never need the fetch");
        auto det = detect_protocol(v4, 28, 90, dirs);
        CHECK(det.tag.rfind("10x-3p-v4", 0) != 0 && det.tag.rfind("10x-5p-v4", 0) != 0,
              "detection never calls GEM-X without its list (got " + det.tag + ")");
    }
    write_list(wl + "/3M-3pgex-may-2023.txt", L4);
    write_list(wl + "/3M-5pgex-jan-2023.txt", L5);
    {   // GEM-X: lists installed
        auto v4 = make_spots(L4, 12, "3p", 16);
        auto det = detect_protocol(v4, 28, 90, dirs);
        CHECK(det.tag == "10x-3p-v4" && det.wl_match_rate > 0.9, "GEM-X 3' reads with the lists installed: 10x-3p-v4 (got " + det.tag + ")");
        auto v5 = make_spots(L5, 12, "5p", 17);
        det = detect_protocol(v5, 41, 90, dirs);
        CHECK(det.tag == "10x-5p-v4", "GEM-X 5' reads: 10x-5p-v4 from its own list (got " + det.tag + ")");
        auto v3 = make_spots(L3, 12, "3p", 18);
        det = detect_protocol(v3, 28, 90, dirs);
        CHECK(det.tag == "10x-3p-v3", "3' v3 reads stay 10x-3p-v3 with the GEM-X lists installed (got " + det.tag + ")");
        CHECK(fetched_whitelist_missing_error("10x-3p-v4", dirs).empty(), "with the list present a GEM-X call proceeds");
        auto junk = make_spots({}, 12, "3p", 21);
        det = detect_protocol(junk, 28, 90, dirs);
        CHECK(det.tag != "10x-3p-v4" && det.tag != "10x-5p-v4",
              "reads of no listed chemistry: the larger GEM-X list does not win the near-zero tie (got " + det.tag + ")");
        auto d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x-3p-v4", v4, dirs);
        CHECK(d.source == "geo" && d.tag == "10x-3p-v4", "LOW v3 call on GEM-X reads, GEO v4, list installed: GEO's GEM-X call passes");
    }
    {   // both pass: a second directory whose GEM-X list is a copy of the v3 list
        char t2[] = "/tmp/singlet-confpol2-XXXXXX";
        const std::string wl2 = mkdtemp(t2);
        write_list(wl2 + "/3M-february-2018.txt", L3);
        write_list(wl2 + "/3M-3pgex-may-2023.txt", L3);
        auto v3 = make_spots(L3, 12, "3p", 19);
        auto d = resolve_protocol_call("10x-3p-v3", Confidence::LOW, "10x-3p-v4", v3, {wl2});
        CHECK(d.source == "excluded" && d.exclusion_reason.find("both pass") != std::string::npos,
              "LOW call and GEO both pass on the same reads: excluded, never guessed");
        for (const char* f : {"/3M-february-2018.txt", "/3M-3pgex-may-2023.txt"}) std::remove((wl2 + f).c_str());
        rmdir(wl2.c_str());
    }
    {   // the fetch pins
        for (const auto& f : fetchable_whitelists()) {
            const std::string sha = f.sha256, url = f.url;
            bool hex = sha.size() == 64;
            for (char c : sha) hex = hex && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
            CHECK(hex && url.rfind("https://storage.googleapis.com/", 0) == 0 && f.bytes > 0,
                  std::string(f.file) + " is pinned by sha256 and fetched over HTTPS");
            CHECK(find_protocol_spec(f.file[3] == '3' ? "10x-3p-v4" : "10x-5p-v4")->whitelist_file == f.file,
                  std::string(f.file) + " is the list of its GEM-X spec");
        }
    }
    {   // 28-nt fallback: no whitelist directory at all still yields a 28-nt 10x call (3' v3, the lowest id of the geometry)
        auto spots = make_spots(L3, 12, "", 20, 500);
        auto det = detect_protocol(spots, 28, 90, std::vector<std::string>{});
        CHECK(det.tag == "10x-3p-v3", "no whitelists: 28-nt R1 falls back to 10x-3p-v3 (got " + det.tag + ")");
    }
    for (const char* f : {"/3M-february-2018.txt", "/737K-august-2016.txt", "/3M-3pgex-may-2023.txt", "/3M-5pgex-jan-2023.txt"})
        std::remove((wl + f).c_str());
    rmdir(wl.c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
