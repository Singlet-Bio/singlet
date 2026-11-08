// SPDX-License-Identifier: MIT
// tests/cpp/test_encode_autodetect_whitelist.cpp
//
// GSE230239 (10x 3' v3, R1 sequenced to 101 nt): singlet encode ran auto-detection without the 10x whitelists because it did not read
// $SINGLET_WHITELIST_DIR, scored the chemistries on read lengths alone, called split-seq and cut R1 to 66 nt. Now:
//  1. auto-detection with no 10x whitelist stops with an error that names --whitelist-dir, $SINGLET_WHITELIST_DIR and <binary>/../whitelists
//  2. with the whitelists, a 101-nt v3-layout R1 is 10x-3p-v3 at HIGH confidence and R1 is cut to 28
//  3. an explicit --protocol needs no whitelist
//  4. resolve_whitelist_dirs: --whitelist-dir first, then $SINGLET_WHITELIST_DIR

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "singlet/fq/fastq_encoder.h"
#include "singlet/fq/reader.h"
#include "test_harness.h"

using namespace singlet::fq;

namespace {

std::string rnd(std::mt19937_64& rng, size_t n) {
    static const char b[] = "ACGT";
    std::string s(n, 'A');
    for (auto& c : s) c = b[rng() % 4];
    return s;
}

std::string make_whitelists(std::vector<std::string>& v3) {
    char tmpl[] = "/tmp/test_encwl_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    std::mt19937_64 rng(3);
    std::ofstream f3(dir + "/3M-february-2018.txt"), f2(dir + "/737K-august-2016.txt");
    for (int i = 0; i < 20000; ++i) { v3.push_back(rnd(rng, 16)); f3 << v3.back() << "\n"; }
    for (int i = 0; i < 8000; ++i) f2 << rnd(rng, 16) << "\n";
    return dir;
}

// R1 101 nt: whitelist barcode (16) + UMI (12) + poly(T) 30 + random 43; R2 91 nt random
void write_v3_pair(const std::string& p1, const std::string& p2, const std::vector<std::string>& wl, size_t n) {
    std::mt19937_64 rng(9);
    std::ofstream f1(p1), f2(p2);
    for (size_t i = 0; i < n; ++i) {
        const std::string a = wl[rng() % wl.size()] + rnd(rng, 12) + std::string(30, 'T') + rnd(rng, 43), b = rnd(rng, 91);
        f1 << "@r" << i << "\n" << a << "\n+\n" << std::string(a.size(), 'I') << "\n";
        f2 << "@r" << i << "\n" << b << "\n+\n" << std::string(b.size(), 'I') << "\n";
    }
}

EncoderConfig base_cfg(const std::string& out) {
    EncoderConfig cfg;
    cfg.output_path = out;
    cfg.no_dedup = true;
    cfg.no_trim = true;
    cfg.qual_mode = QualMode::NONE;
    return cfg;
}

}  // namespace

int main() {
    std::vector<std::string> v3;
    const std::string wl = make_whitelists(v3);
    const std::string p1 = "tmp_encwl_1.fastq", p2 = "tmp_encwl_2.fastq", out = "tmp_encwl.1fq";
    const size_t n = 3000;
    write_v3_pair(p1, p2, v3, n);

    {   // 1. no whitelist: an error naming the three places
        EncoderConfig cfg = base_cfg(out);
        std::string err;
        try { FastqEncoder enc; enc.encode(p1, p2, cfg); } catch (const std::runtime_error& e) { err = e.what(); }
        CHECK(!err.empty(), "auto-detection without whitelists stops with an error");
        CHECK(err.find("--whitelist-dir") != std::string::npos && err.find("$SINGLET_WHITELIST_DIR") != std::string::npos &&
              err.find("../whitelists") != std::string::npos, "the error names --whitelist-dir, $SINGLET_WHITELIST_DIR and <binary>/../whitelists");
        EncoderConfig cfg2 = base_cfg(out);
        cfg2.whitelist_dirs = {"/nonexistent-singlet-wl"};
        err.clear();
        try { FastqEncoder enc; enc.encode(p1, p2, cfg2); } catch (const std::runtime_error& e) { err = e.what(); }
        CHECK(err.find("/nonexistent-singlet-wl") != std::string::npos, "a directory without the 10x whitelists also stops, and is named");
        std::remove(out.c_str());
    }
    {   // 2. with whitelists: 10x-3p-v3 at HIGH, R1 cut to 28
        EncoderConfig cfg = base_cfg(out);
        cfg.whitelist_dirs = {wl};
        FastqEncoder enc;
        const auto st = enc.encode(p1, p2, cfg);
        CHECK(st.protocol_tag == "10x-3p-v3", "101-nt v3-layout R1 with whitelists: 10x-3p-v3 (got " + st.protocol_tag + ")");
        CHECK(st.confidence == Confidence::HIGH, "detected at HIGH confidence");
        CHECK(st.total_reads == n, "every read is encoded");
        Reader rd;
        rd.open(out.c_str());
        CHECK(rd.header().stream_lengths[0] == 28, "archive R1 length is 28");
        rd.close();
        std::remove(out.c_str());
    }
    {   // 3. explicit protocol, no whitelists
        EncoderConfig cfg = base_cfg(out);
        cfg.protocol_tag = "10x-3p-v3";
        bool ok = true;
        try { FastqEncoder enc; const auto st = enc.encode(p1, p2, cfg); ok = st.total_reads == n; } catch (...) { ok = false; }
        CHECK(ok, "--protocol 10x-3p-v3 encodes without whitelists");
        std::remove(out.c_str());
    }
    {   // 4. resolver order
        CHECK(resolve_whitelist_dirs({"/cli-dir"}) == std::vector<std::string>{"/cli-dir"}, "--whitelist-dir wins over the environment");
        setenv("SINGLET_WHITELIST_DIR", wl.c_str(), 1);
        CHECK(resolve_whitelist_dirs({}) == std::vector<std::string>{wl}, "$SINGLET_WHITELIST_DIR is used when no --whitelist-dir is given");
        CHECK(autodetect_whitelists_present(resolve_whitelist_dirs({})), "the environment's directory holds the 10x whitelists");
        unsetenv("SINGLET_WHITELIST_DIR");
    }
    for (const auto& f : {p1, p2}) std::remove(f.c_str());
    for (const char* f : {"/3M-february-2018.txt", "/737K-august-2016.txt"}) std::remove((wl + f).c_str());
    rmdir(wl.c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
