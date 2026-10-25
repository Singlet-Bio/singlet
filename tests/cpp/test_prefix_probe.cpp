// Constant 5' prefix probe (include/singlet/fq/prefix_probe.h): no clip on random cDNA, on barcode-sorted duplicates
// at the start of the file, or on poly(A); clip on a real constant prefix spread through the file.
#include "singlet/fq/prefix_probe.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(c, msg)                                              \
    do {                                                           \
        if (c) { ++g_pass; } else { ++g_fail; std::printf("FAIL: %s\n", msg); } \
    } while (0)

static std::string rnd(std::mt19937& r, int n) {
    static const char B[] = "ACGT";
    std::string s(n, 'A');
    for (auto& c : s) c = B[r() & 3];
    return s;
}

int main() {
    using namespace singlet::fq;
    std::mt19937 r(7);
    // 1. random cDNA: no clip
    {
        std::vector<std::string> v;
        for (int i = 0; i < 12000; ++i) v.push_back(rnd(r, 91));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0, "random reads not clipped");
    }
    // 2. PBMC failure mode: the first reads are duplicates of one molecule (barcode-sorted file), the rest random.
    //    With only the first 25 reads (the old probe) this looked like a 91 bp constant prefix.
    {
        std::vector<std::string> v;
        std::string dup = rnd(r, 91);
        for (int i = 0; i < 25; ++i) v.push_back(dup);
        auto old_like = detect_constant_prefix(v, 0.75, 11, 10);
        CHECK(old_like.raw_prefix_len >= 11, "duplicates alone look constant (old failure reproduced)");
        for (int i = 0; i < 12000; ++i) v.push_back(rnd(r, 91));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0, "duplicates at file start + spread sample: no clip");
    }
    // 3. too few reads: no clip
    {
        std::vector<std::string> v;
        for (int i = 0; i < 200; ++i) v.push_back("ACGTACGTACGTACGTACGTAC" + rnd(r, 60));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0 && p.reason == "too_few_reads", "under 1000 reads: no clip");
    }
    // 4. real 50 bp constant prefix on 95% of reads (SRR34789664-like): clip 50
    {
        std::string pre = rnd(r, 50);
        std::vector<std::string> v;
        for (int i = 0; i < 12000; ++i) v.push_back((r() % 100 < 95 ? pre : rnd(r, 50)) + rnd(r, 50));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 50, "50 bp constant prefix clipped");
    }
    // 5. prefix only in the first half of the file (a block of one library): no clip
    {
        std::string pre = rnd(r, 30);
        std::vector<std::string> v;
        for (int i = 0; i < 6000; ++i) v.push_back(pre + rnd(r, 60));
        for (int i = 0; i < 6000; ++i) v.push_back(rnd(r, 90));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0, "prefix in one half only: no clip");
    }
    // 6. poly(A) reads: no clip
    {
        std::vector<std::string> v;
        for (int i = 0; i < 12000; ++i) v.push_back(std::string(30, 'A') + rnd(r, 60));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0 && p.reason == "poly_a_or_t", "poly(A) prefix: no clip");
    }
    // 7. TSO GGG (3 bp): no clip
    {
        std::vector<std::string> v;
        for (int i = 0; i < 12000; ++i) v.push_back("GGG" + rnd(r, 88));
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0, "3 bp GGG: no clip");
    }
    // 8. FASTQ spread sampler on a temp file (barcode-sorted duplicates first, then random)
    {
        const char* path = "test_prefix_probe.fastq";
        FILE* f = std::fopen(path, "wb");
        std::string dup = rnd(r, 91);
        for (int i = 0; i < 40000; ++i) {
            std::string s = i < 2000 ? dup : rnd(r, 91);
            std::fprintf(f, "@r%d\n%s\n+\n%s\n", i, s.c_str(), std::string(91, 'F').c_str());
        }
        std::fclose(f);
        auto v = sample_fastq_spread(path, 1000, 12);
        CHECK(v.size() >= 10000, "sampler returns >= 10k reads");
        bool all91 = true;
        for (auto& s : v) all91 = all91 && s.size() == 91;
        CHECK(all91, "sampler returns whole sequence lines");
        auto p = detect_constant_prefix(v);
        CHECK(p.prefix_len == 0, "sampled file with duplicate block: no clip");
        std::remove(path);
    }
    std::printf("test_prefix_probe: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
