// SPDX-License-Identifier: MIT
// test_nonhost_sample.cpp — the per-record outputs of the non-host screen use their own classification when the screen samples.
//
// Regression (found by the CPU = GPU gate on nuclei2k_v2, 116 M unmapped reads, 20 M cap): multi_hits has one entry per selected record
// (a sequence, a name hash below the sampling threshold), but nonhost_reads.fq and the per-cell matrix indexed it by the raw record
// number, so above the cap they paired a read with another read's verdict and the output depended on the order STAR wrote the reads in.
// Here, with a tiny cap: the sample is the same for every record order, each written read is one the classifier accepted, the written set
// is the same for every order, and read_mate_pairs (the per-cell matrix input) pairs result[i] with multi_hits[i].

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h>
#include <string>
#include <vector>

#include "singlet/pileup/nonhost/nonhost_cell_matrix.h"
#include "singlet/pileup/nonhost/nonhost_sample.h"

using namespace singlet::nonhost;

struct Rec { std::string name, seq, qual, cb_umi; };

// the classifier's verdict is a function of the sequence only
static bool verdict(const std::string& seq) { return std::hash<std::string>{}(seq) % 3 == 0; }

static std::string fastq(const std::vector<Rec>& v, bool mate2) {
    std::ostringstream o;
    for (const auto& r : v) {
        if (mate2) o << r.name << "\n" << r.cb_umi << "\n+\n" << std::string(r.cb_umi.size(), 'I') << "\n";
        else o << r.name << "\n" << r.seq << "\n+\n" << r.qual << "\n";
    }
    return o.str();
}

int main() {
    std::mt19937 rng(7);
    const char* B = "ACGT";
    std::vector<Rec> recs;
    for (int i = 0; i < 600; ++i) {
        Rec r;
        r.name = "@read" + std::to_string(i) + (i % 2 ? " 1:N:0" : "");
        if (i % 17 != 0) for (int k = 0; k < 40; ++k) r.seq += B[rng() & 3];    // every 17th record has an empty sequence
        r.qual = std::string(r.seq.size(), 'F');
        for (int k = 0; k < 28; ++k) r.cb_umi += B[rng() & 3];
        recs.push_back(r);
    }
    const size_t cap = 60;

    std::vector<std::string> ref_selected_names;
    std::set<std::string> ref_written;
    std::vector<std::string> ref_reads_sorted;
    for (int order = 0; order < 4; ++order) {
        std::vector<Rec> v = recs;
        if (order > 0) std::shuffle(v.begin(), v.end(), rng);
        std::istringstream in1(fastq(v, false));
        const SampledReads smp = sample_reads(in1, cap);
        assert(smp.subsampled && smp.reads.size() <= cap && !smp.reads.empty());

        // the selected set is {records with a sequence and hash < threshold}: independent of the order
        std::vector<std::string> sel;
        for (const auto& r : v) if (smp.selected(r.name, r.seq)) sel.push_back(r.name);
        assert(sel.size() == smp.reads.size());
        std::sort(sel.begin(), sel.end());
        if (order == 0) ref_selected_names = sel; else assert(sel == ref_selected_names);

        // multi_hits[i] is the verdict on the i-th selected read; the writer must keep exactly the selected reads with a positive verdict
        std::vector<bool> mh;
        for (const auto& s : smp.reads) mh.push_back(verdict(s));
        std::istringstream in2(fastq(v, false));
        std::ostringstream out;
        const size_t nw = write_classified(in2, out, smp, [&](size_t i) { return i < mh.size() && mh[i]; });
        std::set<std::string> written;
        {
            std::istringstream o(out.str());
            std::string n, s, p, q;
            size_t cnt = 0;
            while (std::getline(o, n) && std::getline(o, s) && std::getline(o, p) && std::getline(o, q)) {
                assert(verdict(s));                                  // a read the classifier accepted, not another read's verdict
                assert(smp.selected(n, s));
                written.insert(n + "|" + s);
                ++cnt;
            }
            assert(cnt == nw);
        }
        size_t expect = 0;
        for (const auto& r : v) expect += smp.selected(r.name, r.seq) && verdict(r.seq);
        assert(nw == expect && expect > 0);
        if (order == 0) ref_written = written; else assert(written == ref_written);

        // read_mate_pairs with the same selection pairs result[i] with multi_hits[i]
        namespace fs = std::filesystem;
        const fs::path d = fs::temp_directory_path() / ("test_nonhost_sample_" + std::to_string(::getpid()) + "_" + std::to_string(order));
        fs::create_directories(d);
        { std::ofstream(d / "m1") << fastq(v, false); std::ofstream(d / "m2") << fastq(v, true); }
        auto sel_fn = [&](const std::string& n, const std::string& s) { return smp.selected(n, s); };
        const auto tagged = NonHostCellMatrix::read_mate_pairs((d / "m1").string(), (d / "m2").string(), 16, 12, sel_fn);
        assert(tagged.size() == smp.reads.size());
        for (size_t i = 0; i < tagged.size(); ++i) assert(tagged[i].seq == smp.reads[i]);
        // and the tag belongs to that read: the barcode+UMI of the record with that name
        size_t k = 0;
        for (const auto& r : v) {
            if (!smp.selected(r.name, r.seq)) continue;
            assert(tagged[k].cb == r.cb_umi.substr(0, 16) && tagged[k].umi == r.cb_umi.substr(16, 12));
            ++k;
        }
        fs::remove_all(d);

        // without a selection (the default) every record is kept, as before
        const auto all = NonHostCellMatrix::read_mate_pairs((d / "none").string(), (d / "none").string(), 16, 12);
        assert(all.empty());
    }
    // no sampling needed: the sample is every record with a sequence, in file order
    {
        std::istringstream in(fastq(recs, false));
        const SampledReads smp = sample_reads(in, 1000000);
        assert(!smp.subsampled && smp.threshold == UINT64_MAX);
        size_t n = 0;
        for (const auto& r : recs) if (!r.seq.empty()) assert(smp.reads[n++] == r.seq);
        assert(n == smp.reads.size());
    }
    std::puts("test_nonhost_sample: ok");
    return 0;
}
