// D-101 all-site allele counts (allsite.h) on hand-made alignment records and a three-contig STAR-style genome:
// discovery thresholds (molecules, barcodes), the R1 vote per molecule, the low-quality mask, chrM and MAPQ exclusion, reverse-strand
// reads, the best of two alternate bases, and that the result does not depend on read order or thread count.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "singlet/pileup/allsite.h"

namespace A = singlet::allsite;
static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL " << msg << "\n"; ++n_fail; } } while (0)

struct Sig { uint32_t aln_off; uint8_t n_aln; uint8_t nh; uint16_t rlen; uint64_t seq_off; };
struct Aln { int32_t tid; int32_t pos; uint32_t ops_off; uint16_t n_ops; uint8_t flags; uint8_t mapq; };
struct Rd { uint32_t sig; int32_t bc; uint64_t umi; };

struct Recs {
    std::vector<Sig> S; std::vector<Aln> Al; std::vector<Rd> R; std::vector<uint32_t> ops; std::vector<uint8_t> seq; std::vector<uint64_t> lq;
    int qw = 2;
    uint64_t n_sigs() const { return S.size(); }
    uint64_t n_reads() const { return R.size(); }
    const Sig& sig(uint64_t i) const { return S[i]; }
    const Aln& aln(uint64_t i) const { return Al[i]; }
    const Rd& read(uint64_t i) const { return R[i]; }
    const uint32_t* ops_at(uint64_t o) const { return ops.data() + o; }
    const uint8_t* seq_at(uint64_t o) const { return seq.data() + o; }
    uint64_t lowq_word(uint64_t i) const { return lq[i]; }
    // a signature: sequence (original orientation, ACGT), one primary alignment with a single M block
    uint32_t add_sig(const std::string& s, int32_t tid, int32_t pos, bool rev, uint8_t mapq = 255) {
        Sig g{static_cast<uint32_t>(Al.size()), 1, 1, static_cast<uint16_t>(s.size()), seq.size()};
        auto code = [](char c) -> uint8_t { return c == 'A' ? 1 : c == 'C' ? 2 : c == 'G' ? 4 : c == 'T' ? 8 : 15; };
        for (size_t i = 0; i < s.size(); i += 2) seq.push_back(static_cast<uint8_t>(code(s[i]) | ((i + 1 < s.size() ? code(s[i + 1]) : 0) << 4)));
        Al.push_back({tid, pos, static_cast<uint32_t>(ops.size()), 1, static_cast<uint8_t>(2 | (rev ? 1 : 0)), mapq});
        ops.push_back((static_cast<uint32_t>(s.size()) << 4) | 0);
        S.push_back(g); return static_cast<uint32_t>(S.size() - 1);
    }
    void add_read(uint32_t sig, int32_t bc, const std::string& umi, uint32_t lowq_bit = UINT32_MAX) {
        uint64_t u = 0; for (size_t i = 0; i < umi.size(); ++i) u |= static_cast<uint64_t>(umi[i] == 'A' ? 0 : umi[i] == 'C' ? 1 : umi[i] == 'G' ? 2 : 3) << (3 * i);
        R.push_back({sig, bc, u});
        lq.push_back(0); lq.push_back(0);
        if (lowq_bit != UINT32_MAX) lq[lq.size() - 2 + (lowq_bit >> 6)] |= 1ull << (lowq_bit & 63);
    }
};

static std::string revcomp(std::string s) {
    std::reverse(s.begin(), s.end());
    for (char& c : s) c = c == 'A' ? 'T' : c == 'C' ? 'G' : c == 'G' ? 'C' : c == 'T' ? 'A' : 'N';
    return s;
}

static uint32_t at(const A::Result& r, const singlet::SparseAccumulator<uint16_t>::CSCMatrix& m, uint32_t row, uint32_t col) {
    (void) r;
    for (int32_t k = m.indptr[col]; k < m.indptr[col + 1]; ++k) if (static_cast<uint32_t>(m.indices[k]) == row) return m.data[k];
    return 0;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("allsite_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    // genome: chr1 (64 bases), chrM (32), chr2 (64); STAR pads contigs to 2^bin; here chrStart is explicit
    std::string chr1, chr2, chrM(32, 'A');
    { std::mt19937 g(7); const char* b = "ACGT"; for (int i = 0; i < 64; ++i) { chr1 += b[g() % 4]; chr2 += b[g() % 4]; } }
    { std::ofstream f(dir / "chrName.txt"); f << "chr1\nchrM\nchr2\n"; }
    { std::ofstream f(dir / "chrStart.txt"); f << "0\n64\n128\n192\n"; }
    { std::ofstream f(dir / "chrNameLength.txt"); f << "chr1\t64\nchrM\t32\nchr2\t64\n"; }
    { std::ofstream f(dir / "Genome", std::ios::binary); auto put = [&](const std::string& s, size_t pad) {
          for (char c : s) f.put(static_cast<char>(c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : 3)); for (size_t i = s.size(); i < pad; ++i) f.put(5); };
      put(chr1, 64); put(chrM, 64); put(chr2, 64); }
    const std::vector<std::string> tids = {"chr1", "chrM", "chr2"};
    A::Genome G; std::string err;
    CHECK(G.open(dir.string(), tids, &err), "genome open " + err);
    CHECK(G.base(0, 0) == std::string("ACGT").find(chr1[0]), "genome base");
    CHECK(G.base(2, 63) == std::string("ACGT").find(chr2[63]), "genome base chr2");
    CHECK(G.base(0, 64) == 4, "outside the contig");

    auto other = [](char c, int k) { const std::string b = "ACGT"; std::string o; for (char x : b) if (x != c) o += x; return o[k]; };
    Recs R;
    // chr1 site at pos 10: ALT = other(ref,0). Reads cover [5, 25).
    const std::string ref1 = chr1.substr(5, 20);
    std::string alt1 = ref1; alt1[5] = other(ref1[5], 0);           // pos 10 ALT
    std::string alt1b = ref1; alt1b[5] = other(ref1[5], 1);         // pos 10 second ALT
    std::string alt20 = ref1; alt20[15] = other(ref1[15], 0);       // pos 20 ALT (one barcode only)
    const uint32_t sR = R.add_sig(ref1, 0, 5, false), sA = R.add_sig(alt1, 0, 5, false), sB = R.add_sig(alt1b, 0, 5, false), sC = R.add_sig(alt20, 0, 5, false);
    const uint32_t sArev = R.add_sig(revcomp(alt1), 0, 5, true);    // same alignment, reverse strand, record stored in read orientation
    // barcode 0: molecule AAAA = 2 ALT reads + 1 REF read (vote ALT); molecule CCCC = REF
    R.add_read(sA, 0, "AAAA"); R.add_read(sA, 0, "AAAA"); R.add_read(sR, 0, "AAAA"); R.add_read(sR, 0, "CCCC");
    // barcode 1: molecule GGGG ALT via the reverse-strand record; molecule TTTT tie (1 ALT, 1 REF) -> ambiguous
    R.add_read(sArev, 1, "GGGG"); R.add_read(sA, 1, "TTTT"); R.add_read(sR, 1, "TTTT");
    // barcode 2: ALT read masked low quality at record index 5 -> only its REF molecule counts; second ALT base, one molecule
    R.add_read(sA, 2, "ACAC", 5); R.add_read(sR, 2, "GTGT"); R.add_read(sB, 2, "CATG");
    // barcode 3: two molecules with the ALT at pos 20 (one barcode: not a site), plus the second ALT at pos 10 in a second barcode
    R.add_read(sC, 3, "AAAA"); R.add_read(sC, 3, "CCCC"); R.add_read(sB, 3, "GGGG");
    // chrM and low MAPQ: an ALT carried by many barcodes, excluded
    const std::string mref = chrM.substr(0, 16); std::string malt = mref; malt[3] = 'C';
    const uint32_t sM = R.add_sig(malt, 1, 0, false);
    for (int b = 0; b < 4; ++b) R.add_read(sM, b, "AAAA");
    const std::string ref2 = chr2.substr(30, 20); std::string alt2 = ref2; alt2[2] = other(ref2[2], 2);
    const uint32_t sL = R.add_sig(alt2, 2, 30, false, 3);
    for (int b = 0; b < 4; ++b) R.add_read(sL, b, "AAAA");
    // barcode -1 (not in the set): ignored even with an ALT
    R.add_read(sC, -1, "TTTT");

    A::Params P; P.skip_tid = 1; P.use_qual = 1; P.qwords = 2; P.umi_len = 4; P.threads = 1;
    const A::Result r = A::run(R, G, 3, 4, P);
    CHECK(r.sites.size() == 1, "one site, got " + std::to_string(r.sites.size()));
    if (r.sites.size() == 1) {
        const A::Site& s = r.sites[0];
        CHECK(s.tid == 0 && s.pos == 10, "site at chr1:11");
        CHECK(s.ref == ref1[5] && s.alt == alt1[5], "site alleles (first ALT has more molecules)");
        // ALT molecules at discovery: bc0 AAAA, bc1 GGGG, bc1 TTTT (exact UMIs; the masked read is out) = 3 in 2 barcodes
        CHECK(s.alt_molecules == 3 && s.alt_cells == 2, "discovery counts " + std::to_string(s.alt_molecules) + "/" + std::to_string(s.alt_cells));
        CHECK(A::site_name(s, tids) == "chr1:11:" + std::string(1, s.ref) + ">" + std::string(1, s.alt), "site name");
        // DP / AD per barcode: bc0 2/1, bc1 1/1 (tie dropped), bc2 1/0 (masked ALT; second-ALT molecule is "other"), bc3 2/0 (two REF molecules from the pos-20 reads)
        CHECK(at(r, r.dp, 0, 0) == 2 && at(r, r.ad, 0, 0) == 1, "bc0");
        CHECK(at(r, r.dp, 0, 1) == 1 && at(r, r.ad, 0, 1) == 1, "bc1");
        CHECK(at(r, r.dp, 0, 2) == 1 && at(r, r.ad, 0, 2) == 0, "bc2");
        CHECK(at(r, r.dp, 0, 3) == 2 && at(r, r.ad, 0, 3) == 0, "bc3");
        CHECK(r.ambiguous == 1, "one tie");
        CHECK(r.other == 2, "two other-base molecules, got " + std::to_string(r.other));
    }
    // order and threads: shuffled reads, 3 threads, same matrices
    Recs R2 = R; std::mt19937 g(11);
    { std::vector<size_t> idx(R.R.size()); for (size_t i = 0; i < idx.size(); ++i) idx[i] = i; std::shuffle(idx.begin(), idx.end(), g);
      R2.R.clear(); R2.lq.clear(); for (size_t i : idx) { R2.R.push_back(R.R[i]); R2.lq.push_back(R.lq[2 * i]); R2.lq.push_back(R.lq[2 * i + 1]); } }
    P.threads = 3;
    const A::Result r2 = A::run(R2, G, 3, 4, P);
    CHECK(r2.sites.size() == r.sites.size() && r2.dp.data == r.dp.data && r2.dp.indices == r.dp.indices && r2.dp.indptr == r.dp.indptr
          && r2.ad.data == r.ad.data && r2.ad.indices == r.ad.indices, "order and thread count do not change the result");
    // thresholds: 1 molecule in 1 barcode admits pos 20 and the second ALT stays unchosen at pos 10
    P.min_alt_molecules = 1; P.min_alt_cells = 1;
    const A::Result r3 = A::run(R, G, 3, 4, P);
    CHECK(r3.sites.size() == 2, "two sites at 1/1, got " + std::to_string(r3.sites.size()));
    // sites.tsv
    const std::string tsv = (dir / "sites.tsv.gz").string();
    CHECK(A::write_sites_tsv(tsv, r, tids), "sites.tsv written");
    fs::remove_all(dir);
    std::cout << (n_fail ? "FAILED " : "OK ") << n_fail << "\n";
    return n_fail ? 1 : 0;
}
