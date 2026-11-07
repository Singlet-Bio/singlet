// Unit test for order-free-rules R11 (CbCorrector: STAR's 1MM_multi_Nbase_pseudocounts CB correction).
// Build: g++ -O2 -std=c++17 -Iinclude tests/test_cb_correct.cpp -o test_cb_correct && ./test_cb_correct
#include <cstdio>
#include <fstream>
#include <string>
#include "singlet/pileup/cb_correct.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

int main() {
    const char* wl_path = "test_cb_wl.txt";
    {
        std::ofstream f(wl_path);
        f << "AAAAAAAA\nAAAAAAAC\nAAAAAAAG\nCCCCCCCC\nGGGGGGGG\nNGGGGGGG\nTTTTTTTT\nTTTTTTTT\n";   // N at 0 skipped, dup removed
    }
    singlet::CbCorrector c;
    CHECK(c.load_whitelist(wl_path));
    CHECK(c.size() == 6);
    CHECK(c.length() == 8);
    char out[9] = {};
    auto res = [&](const char* cb) { out[8] = '\0'; return c.resolve(cb, nullptr, out); };

    CHECK(res("CCCCCCCC") == singlet::CbCorrector::kExact); CHECK(std::string(out) == "CCCCCCCC");
    CHECK(res("CCCCCCCA") == singlet::CbCorrector::kOneCandidate); CHECK(std::string(out) == "CCCCCCCC");
    CHECK(res("ACACACAC") == singlet::CbCorrector::kNoMatch);
    CHECK(res("NNCCCCCC") == singlet::CbCorrector::kTwoN);
    CHECK(res("CCCNCCCC") == singlet::CbCorrector::kOneCandidate); CHECK(std::string(out) == "CCCCCCCC");
    // AAAAAAAT is one base from AAAAAAAA, AAAAAAAC and AAAAAAAG: with no exact counts all three have 1 (pseudocount),
    // posterior 1/3 each -> rejected
    CHECK(res("AAAAAAAT") == singlet::CbCorrector::kRejected);
    // one N at the last base: candidates A, C, G (T absent), also rejected without counts
    CHECK(res("AAAAAAAN") == singlet::CbCorrector::kRejected);
    // give AAAAAAAC 100 exact reads: 101 / (1 + 101 + 1) = 0.9806 >= 0.975 -> accepted
    c.add_exact(c.find("AAAAAAAC"), 100);
    CHECK(res("AAAAAAAT") == singlet::CbCorrector::kPosterior); CHECK(std::string(out) == "AAAAAAAC");
    CHECK(res("AAAAAAAN") == singlet::CbCorrector::kPosterior); CHECK(std::string(out) == "AAAAAAAC");
    // 77 reads: 78 / 80 = 0.975 exactly in real numbers; float decides (STAR's semantics), check the neighbourhood
    singlet::CbCorrector d;
    d.load_whitelist(wl_path);
    d.add_exact(d.find("AAAAAAAC"), 70);   // 71 / 73 = 0.9726 < 0.975 -> rejected
    CHECK(d.resolve("AAAAAAAT", nullptr, out) == singlet::CbCorrector::kRejected);
    singlet::CbCorrector e;
    e.load_whitelist(wl_path);
    e.add_exact(e.find("AAAAAAAC"), 90);   // 91 / 93 = 0.9785 -> accepted
    CHECK(e.resolve("AAAAAAAT", nullptr, out) == singlet::CbCorrector::kPosterior);
    // ties: equal counts, first candidate in STAR's order wins pmax but the posterior is 1/3 -> rejected anyway
    std::remove(wl_path);
    std::printf(failures ? "R11 tests: %d FAILED\n" : "R11 tests: all passed%.0d\n", failures);
    return failures ? 1 : 0;
}
