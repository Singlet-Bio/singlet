// SPDX-License-Identifier: MIT
// test_nonhost_fastscan.cpp — the parallel passes over the unmapped-read files (nonhost_fastscan.h) give what the serial readers give.
//
// For synthetic mate1 and mate2 files (empty sequences, quality lines that start with '@', carriage returns, a last record without its
// final newline, a truncated last record) and several caps and thread counts:
//   fast_sample == sample_reads (reads, final threshold, subsampled), fast_write_classified == write_classified (bytes and count),
//   fast_tag_reads == NonHostCellMatrix::read_mate_pairs with the same selection.
// A file whose layout is not the 4-line one must be refused (the caller then runs the serial reader).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "singlet/pileup/nonhost/nonhost_cell_matrix.h"
#include "singlet/pileup/nonhost/nonhost_fastscan.h"
#include "singlet/pileup/nonhost/nonhost_sample.h"

using namespace singlet::nonhost;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Files { std::string m1, m2; };

// n records; tail: 0 = complete, 1 = last record without the final newline, 2 = a truncated last record (three lines)
static Files make_files(const std::string& dir, const std::string& tag, size_t n, int tail, unsigned seed, bool crlf) {
    std::mt19937 rng(seed);
    const char* B = "ACGT";
    const std::string nl = crlf ? "\r\n" : "\n";
    std::string a, b;
    for (size_t i = 0; i < n; ++i) {
        std::string name = "@r" + std::to_string(rng() % 100000000) + "_" + std::to_string(i) + (i % 3 ? " 0:N:  0" : "");
        std::string seq;
        if (i % 19 != 0) { const size_t L = 30 + rng() % 90; for (size_t k = 0; k < L; ++k) seq += B[rng() & 3]; }
        std::string qual(seq.size(), 'F');
        if (!qual.empty() && i % 5 == 0) qual[0] = '@';   // a quality line that looks like a name line
        if (!qual.empty() && i % 7 == 0) qual[0] = '+';
        std::string cu;
        const size_t cl = (i % 23 == 0) ? 10 : 28;   // a barcode read shorter than barcode + UMI now and then
        for (size_t k = 0; k < cl; ++k) cu += B[rng() & 3];
        a += name + nl + seq + nl + "+" + nl + qual + nl;
        b += name + nl + cu + nl + "+" + nl + std::string(cu.size(), 'I') + nl;
    }
    if (tail == 1) { a.erase(a.size() - nl.size()); b.erase(b.size() - nl.size()); }
    if (tail == 2) { a += "@trunc" + nl + "ACGT" + nl + "+" + nl; }
    Files f{dir + "/" + tag + ".m1", dir + "/" + tag + ".m2"};
    std::ofstream(f.m1, std::ios::binary) << a;
    std::ofstream(f.m2, std::ios::binary) << b;
    return f;
}

static void compare(const Files& f, size_t cap, int threads, const std::string& dir) {
    // serial references
    SampledReads ref;
    { std::ifstream in(f.m1); ref = sample_reads(in, cap); }
    SampledReads got;
    fastscan::FastIndex fi;
    const bool ok = fastscan::fast_sample(f.m1, cap, threads, got, fi);
    CHECK(ok);
    if (!ok) return;
    CHECK(got.reads == ref.reads);
    CHECK(got.threshold == ref.threshold);
    CHECK(got.subsampled == ref.subsampled);
    CHECK(fi.size() == ref.reads.size());

    // the classified-read writer
    auto keep = [&](size_t i) { return std::hash<std::string>{}(ref.reads[i]) % 3 != 0; };
    std::ostringstream so;
    size_t nw_ref;
    { std::ifstream in(f.m1); nw_ref = write_classified(in, so, ref, keep); }
    const std::string out_path = dir + "/classified.fq";
    size_t nw_got = 0;
    CHECK(fastscan::fast_write_classified(fi, keep, threads, out_path, &nw_got));
    CHECK(nw_got == nw_ref);
    CHECK(slurp(out_path) == so.str());

    // barcode and UMI from mate2
    auto sel = [&](const std::string& name_l, const std::string& seq_l) { return ref.selected(name_l, seq_l); };
    const auto tr_ref = NonHostCellMatrix::read_mate_pairs(f.m1, f.m2, 16, 12, sel);
    std::vector<TaggedRead> tr_got;
    CHECK(fastscan::fast_tag_reads(f.m2, fi, got.reads, 16, 12, threads, tr_got));
    CHECK(tr_got.size() == tr_ref.size());
    if (tr_got.size() == tr_ref.size())
        for (size_t i = 0; i < tr_ref.size(); ++i)
            if (tr_got[i].seq != tr_ref[i].seq || tr_got[i].cb != tr_ref[i].cb || tr_got[i].umi != tr_ref[i].umi) { CHECK(false); break; }
}

int main() {
    const std::string dir = (std::filesystem::temp_directory_path() / ("nhfs_" + std::to_string(getpid()))).string();
    std::filesystem::create_directories(dir);
    int n = 0;
    for (size_t recs : {size_t(0), size_t(1), size_t(7), size_t(300), size_t(40000)})
        for (int tail = 0; tail < 3; ++tail)
            for (bool crlf : {false, true}) {
                if (recs == 0 && (tail || crlf)) continue;
                if (recs < 3 && tail == 2 && recs == 0) continue;
                const Files f = make_files(dir, "t" + std::to_string(n++), recs, tail, 11 + static_cast<unsigned>(recs), crlf);
                for (size_t cap : {size_t(20000000), size_t(5000), size_t(97), size_t(1)})
                    for (int threads : {1, 3, 8}) {
                        if (recs < 300 && threads > 1) continue;
                        compare(f, cap, threads, dir);
                    }
            }

    // not the 4-line layout: refused
    {
        const std::string p = dir + "/bad.m1";
        std::ofstream(p) << "@a\nACGT\n+\nFFFF\nNOT A RECORD\nACGT\n+\nFFFF\n";
        SampledReads s;
        fastscan::FastIndex fi;
        CHECK(!fastscan::fast_sample(p, 100, 2, s, fi));
    }
    std::filesystem::remove_all(dir);
    if (g_fail) { std::fprintf(stderr, "%d checks failed\n", g_fail); return 1; }
    std::printf("test_nonhost_fastscan: ok\n");
    return 0;
}
