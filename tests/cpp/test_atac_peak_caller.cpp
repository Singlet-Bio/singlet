// SPDX-License-Identifier: MIT
// test_atac_peak_caller.cpp — unit tests for atac_peak_caller.h (insertion-based MACS2-style peak caller).
// Synthetic insertions only; no files besides a temporary BED.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "singlet/pileup/atac_peak_caller.h"

namespace {

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 33);
    }
    // uniform integer in [0, n)
    uint32_t below(uint32_t n) { return next() % n; }
};

// -log10 P(X >= k) by a direct forward sum of the pmf in long double.
double reference_tail(unsigned k, double lambda) {
    long double term = std::exp(-static_cast<long double>(lambda));
    for (unsigned i = 1; i <= k; ++i) term *= static_cast<long double>(lambda) / i;
    long double sum = 0;
    for (unsigned i = k; i < k + 2000; ++i) {
        sum += term;
        term *= static_cast<long double>(lambda) / (i + 1);
    }
    return static_cast<double>(-std::log10(sum));
}

void test_poisson_tail() {
    const struct { unsigned k; double lambda; } cases[] = {{10, 2.0}, {50, 5.0}, {8, 0.3}, {30, 12.0}, {3, 0.5}};
    for (const auto& c : cases) {
        const double got = singlet::detail::poisson_upper_neg_log10(c.k, c.lambda);
        const double want = reference_tail(c.k, c.lambda);
        assert(std::fabs(got - want) <= 1e-6 * std::fabs(want));
    }
    assert(singlet::detail::poisson_upper_neg_log10(0, 3.0) == 0.0);
    assert(singlet::detail::poisson_upper_neg_log10(5, 0.0) == 300.0);
    std::printf("  poisson_tail: PASS\n");
}

struct Planted { int chrom; int32_t centre; };

void test_synthetic_genome(std::vector<singlet::AtacPeak>& peaks_out, std::vector<std::string>& names_out) {
    const int32_t L = 2000000;
    const size_t nc = 3;
    std::vector<int32_t> chrom_len(nc, L);
    std::vector<bool> skip = {false, false, true};
    std::vector<std::vector<int32_t>> ins(nc);
    Lcg rng(12345);

    // Uniform background, ~1 insertion per 2 kb.
    for (size_t c = 0; c < nc; ++c)
        for (int i = 0; i < L / 2000; ++i) ins[c].push_back(static_cast<int32_t>(rng.below(static_cast<uint32_t>(L))));

    // Planted dense sites: 300 insertions within +-60 bp of the centre.
    const std::vector<Planted> planted = {{0, 200000}, {0, 700000}, {0, 1300000}, {1, 400000}, {1, 1500000}};
    const Planted on_skipped = {2, 1000000};
    auto plant = [&](const Planted& p) {
        for (int i = 0; i < 300; ++i)
            ins[static_cast<size_t>(p.chrom)].push_back(p.centre - 60 + static_cast<int32_t>(rng.below(121)));
    };
    for (const auto& p : planted) plant(p);
    plant(on_skipped);

    singlet::InsertionPeakParams prm;  // defaults: q 0.05, min_length 200
    auto peaks = singlet::call_peaks_insertions(ins, chrom_len, skip, prm);

    for (const auto& p : planted) {
        int n_in = 0;
        const singlet::AtacPeak* hit = nullptr;
        for (const auto& pk : peaks)
            if (pk.chrom_idx == p.chrom && pk.start <= p.centre && p.centre < pk.end) { ++n_in; hit = &pk; }
        assert(n_in == 1);
        assert(std::abs(hit->summit - p.centre) <= 50);
        assert(hit->pileup > 100.f);
    }
    for (const auto& pk : peaks) assert(pk.chrom_idx != 2);
    assert(peaks.size() <= planted.size() + 2);
    assert(peaks.size() >= planted.size());

    for (size_t i = 0; i < peaks.size(); ++i) {
        const auto& pk = peaks[i];
        assert(pk.end - pk.start >= prm.min_length);
        assert(pk.start <= pk.summit && pk.summit < pk.end);
        assert(pk.neg_log10_q >= 2.0f);  // q <= 0.01
        if (i > 0) {
            const auto& pv = peaks[i - 1];
            assert(pv.chrom_idx < pk.chrom_idx || (pv.chrom_idx == pk.chrom_idx && pv.start < pk.start));
            if (pv.chrom_idx == pk.chrom_idx) assert(pv.end <= pk.start);
        }
    }
    std::printf("  synthetic_genome: PASS (%zu peaks)\n", peaks.size());
    peaks_out = peaks;
    names_out = {"chr1", "chr2", "chrM"};
}

void test_bed(const std::vector<singlet::AtacPeak>& peaks, const std::vector<std::string>& names) {
    const std::string path = "test_atac_peak_caller_tmp.bed";
    assert(singlet::write_insertion_peaks_bed(path, peaks, names));
    std::ifstream in(path);
    assert(in);
    std::string line;
    size_t n = 0;
    while (std::getline(in, line)) {
        size_t tabs = 0;
        for (char ch : line) tabs += (ch == '\t');
        assert(tabs == 9);  // 10 columns
        assert(line.rfind(names[static_cast<size_t>(peaks[n].chrom_idx)] + "\t", 0) == 0);
        ++n;
    }
    assert(n == peaks.size());
    in.close();
    std::remove(path.c_str());
    assert(!singlet::write_insertion_peaks_bed("no_such_dir_xyz/out.bed", peaks, names));
    std::printf("  write_bed: PASS\n");
}

}  // namespace

int main() {
    std::printf("test_atac_peak_caller\n");
    test_poisson_tail();
    std::vector<singlet::AtacPeak> peaks;
    std::vector<std::string> names;
    test_synthetic_genome(peaks, names);
    test_bed(peaks, names);
    std::printf("All tests passed.\n");
    return 0;
}
