// SPDX-License-Identifier: MIT
// tests/cpp/test_chimeric_dedup.cpp
//
// The deduplicated --chimeric second pass: stage A (distinct sequences, representatives, member table) and the expansion of the pass's
// junction lines (name = sequence id) over the members. The same synthetic reads are written once as today's junction file (every read
// mapped, its own name, CB and UMI) and once as the dedup form (one line set per sequence, id as name) with the member table: the rows and
// every Stats counter must be equal, for canonical-only and all-motif calls and for both library orientations. Duplicated sequences,
// multimapping lines, non-canonical and CT/AC lines, a line without barcode columns, exact / one-mismatch / ambiguous / unmatched / N
// barcodes. Also: the representative records, the refusals of stage A, and the self-check against the printed CB and UMI.

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "singlet/pileup/chimeric_dedup.h"
#include "test_harness.h"

using namespace singlet::chimeric;

namespace {
void put(const std::string& path, const std::string& text) { std::ofstream(path) << text; }
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Rd { std::string seq, cb, umi; };

std::string fq(const std::string& name, const std::string& seq, const std::string& qual) { return "@" + name + "\n" + seq + "\n+\n" + qual + "\n"; }

// junction line: donor chr/pos/strand, acceptor chr/pos/strand, type, name, nAln, score, best, cb, umi
std::string jline(const std::string& c1, int p1, const std::string& s1, const std::string& c2, int p2, const std::string& s2, int type,
                  const std::string& name, int n_aln, int score, int best, const std::string& cb, const std::string& umi) {
    return c1 + "\t" + std::to_string(p1) + "\t" + s1 + "\t" + c2 + "\t" + std::to_string(p2) + "\t" + s2 + "\t" + std::to_string(type) + "\t0\t0\t" + name +
           "\t100\t30M\t500\t40M\t" + std::to_string(n_aln) + "\t90\t60\t" + std::to_string(score) + "\t" + std::to_string(best) + "\t0\t" + cb + "\t" + umi + "\n";
}

// the lines a sequence gives, by sequence index, with the name and barcode of the read they are written for
std::string lines_of(int sq, const std::string& name, const std::string& cb, const std::string& umi) {
    std::string o;
    if (sq == 0) {
        o += jline("chr1", 1500, "+", "chr2", 5200, "+", 1, name, 1, 70, 70, cb, umi);      // canonical
        o += jline("chr3", 500, "-", "chr1", 1200, "-", 0, name, 1, 66, 66, cb, umi);       // non-canonical
        o += jline("chr2", 5550, "-", "chr3", 500, "+", 1, name, 2, 60, 64, cb, umi);       // worse alignment of a multimapping read
        o += jline("chr5", 8500, "+", "chr4", 2000, "-", 2, name, 2, 64, 64, cb, umi);      // CT/AC, best: turned around
    } else if (sq == 2) {
        o += "chr2\t1\t+\tchr3\t2\t+\t1\t0\t0\t" + name + "\n";                              // short line: no barcode columns
    } else if (sq == 3) {
        o += jline("chr1", 1500, "+", "chr2", 5200, "+", 1, name, 1, 68, 68, cb, umi);
    }
    return o;
}

const char* kHdr = "chr_donorA\tbrkpt_donorA\tstrand_donorA\tchr_acceptorB\tbrkpt_acceptorB\tstrand_acceptorB\tjunction_type\trepeat_left_lenA\trepeat_right_lenB\t"
                   "read_name\tstart_alnA\tcigar_alnA\tstart_alnB\tcigar_alnB\tnum_chim_aln\tmax_poss_aln_score\tnon_chim_aln_score\tthis_chim_aln_score\t"
                   "bestall_chim_aln_score\tPEmerged_bool\n";

bool same_rows(const std::vector<Row>& a, const std::vector<Row>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        const Row &x = a[i], &y = b[i];
        if (x.barcode != y.barcode || x.donor_chr != y.donor_chr || x.acceptor_chr != y.acceptor_chr || x.donor_strand != y.donor_strand ||
            x.acceptor_strand != y.acceptor_strand || x.donor_pos != y.donor_pos || x.acceptor_pos != y.acceptor_pos || x.junction_type != y.junction_type ||
            x.n_reads != y.n_reads || x.n_umis != y.n_umis || x.n_chim_aln != y.n_chim_aln || x.best_score != y.best_score || x.multimap != y.multimap)
            return false;
    }
    return true;
}
bool same_stats(const Stats& a, const Stats& b) {
    return a.lines == b.lines && a.kept_lines == b.kept_lines && a.no_barcode == b.no_barcode && a.unmatched == b.unmatched && a.ambiguous == b.ambiguous &&
           a.rows == b.rows && a.cells == b.cells && a.annotated == b.annotated && a.noncanonical == b.noncanonical && a.reoriented == b.reoriented &&
           a.blacklisted_junctions == b.blacklisted_junctions && a.blacklisted_rows == b.blacklisted_rows && a.recurrent_rows == b.recurrent_rows;
}
}  // namespace

int main() {
    const std::string dir = "/tmp/test_chimeric_dedup_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    const std::string cbA = "AAACCCAAGAAACACT", cbB = "TTTGGGAAGAAACACT";
    const std::string S[4] = {"ACGTACGTACGTACGTACGTACGTAAAACCCC", "GGGGTTTTACGTACGTACGTAAAACCCCGGGG", "ACGTNACGTACGTACGTACGTTTTTAAAACCC", "TTTTACGTACGTACGTACGTACGTCCCCAAAA"};
    // unmapped file, then soft-clip file (input order)
    const std::vector<Rd> unm = {{S[0], cbA, "AAAAACCCCC"},
                                 {S[1], cbB, "GGGGGTTTTT"},
                                 {S[0], cbA, "AAAAACCCCC"},                          // duplicate of read 0, same UMI
                                 {S[0], "AAACCCAAGAAACACA", "ACACACACAC"},           // two one-mismatch neighbours: ambiguous
                                 {S[2], "TTTGGGAAGAAACACA", "CAAAAAAAAA"},           // one-mismatch neighbour of cell B
                                 {S[1], cbA, "TTTTTAAAAA"}};
    const std::vector<Rd> sc = {{S[0], "CCCCCCCCCCCCCCCC", "ACGTACGTAC"},            // unmatched
                                {S[3], cbA, "GATTACAGAT"},
                                {S[0], "AAACCCAAGAAACACN", "ACGTACGTAA"},            // N: matches two neighbours
                                {S[0], cbB, "AAAAACCCCC"}};
    std::string u1, u2, s1, s2;
    int n = 0;
    for (const auto& r : unm) { u1 += fq("read" + std::to_string(n) + " 0:N:", r.seq, std::string(r.seq.size(), 'I')); u2 += fq("read" + std::to_string(n) + " 1:N:", r.cb + r.umi, std::string(26, 'F')); n++; }
    for (const auto& r : sc) { s1 += fq("read" + std::to_string(n), r.seq, std::string(r.seq.size(), 'J')); s2 += fq("read" + std::to_string(n), r.cb + r.umi, std::string(26, 'G')); n++; }
    put(dir + "/u1", u1); put(dir + "/u2", u2); put(dir + "/s1", s1); put(dir + "/s2", s2);
    DedupParams dp;
    dp.cb_start = 1; dp.cb_len = 16; dp.umi_start = 17; dp.umi_len = 10; dp.barcode_len = 26;

    // ---- stage A ----
    DedupResult R = build_dedup({{dir + "/u1", dir + "/u2"}, {dir + "/s1", dir + "/s2"}}, dir + "/d1", dir + "/d2", dp);
    CHECK(R.ok, "stage A runs: " + R.reason);
    CHECK(R.n_reads == 10 && R.n_first == 6 && R.n_second == 4 && R.n_distinct == 4, "10 reads (6 + 4), 4 distinct sequences");
    CHECK(R.table.n_ids() == 4 && R.table.count(0) == 6 && R.table.count(1) == 2 && R.table.count(2) == 1 && R.table.count(3) == 1, "members per sequence");
    // members in input order, representative first
    std::vector<Rd> all = unm;
    all.insert(all.end(), sc.begin(), sc.end());
    std::map<std::string, int> idof;
    std::vector<std::vector<size_t>> members;
    std::vector<int> seq_of;   // sequence index (into S) of each id
    for (size_t i = 0; i < all.size(); i++) {
        auto it = idof.find(all[i].seq);
        if (it == idof.end()) {
            it = idof.emplace(all[i].seq, (int)members.size()).first;
            members.emplace_back();
            int si = 0;
            for (int k = 0; k < 4; k++) if (S[k] == all[i].seq) si = k;
            seq_of.push_back(si);
        }
        members[it->second].push_back(i);
    }
    bool mem_ok = R.ok;
    for (size_t id = 0; id < members.size() && R.ok; id++) {
        mem_ok = mem_ok && R.table.count(id) == members[id].size();
        for (size_t k = 0; k < members[id].size() && mem_ok; k++) {
            const Rd& r = all[members[id][k]];
            mem_ok = MemberTable::unpack(R.table.cb[R.table.start[id] + k], 16) == r.cb && MemberTable::unpack(R.table.umi[R.table.start[id] + k], 10) == r.umi;
        }
    }
    CHECK(mem_ok, "the member table holds each read's CB and UMI, in input order, representative first");
    // representatives: "@<id>", the first read's sequence and quality, its own barcode record
    std::string e1, e2;
    for (size_t id = 0; id < members.size(); id++) {
        const size_t i = members[id][0];
        e1 += fq(std::to_string(id), all[i].seq, std::string(all[i].seq.size(), i < unm.size() ? 'I' : 'J'));
        e2 += fq(std::to_string(id), all[i].cb + all[i].umi, std::string(26, i < unm.size() ? 'F' : 'G'));
    }
    CHECK(slurp(dir + "/d1") == e1 && slurp(dir + "/d2") == e2, "the representative FASTQ pair");

    // ---- refusals of stage A ----
    {
        put(dir + "/b1", fq("x", "ACGTACGTACGTACGTACGTACGTACGT", std::string(28, 'I')));
        put(dir + "/b2", fq("x", "aaacccaagaaacactAAAAACCCCC", std::string(26, 'F')));
        DedupResult B = build_dedup({{dir + "/b1", dir + "/b2"}}, dir + "/e1", dir + "/e2", dp);
        CHECK(!B.ok && !B.reason.empty(), "a barcode with a lower-case base is refused");
        put(dir + "/b2", fq("x", "AAACCCAAGAAACACTAAAAACCCC", std::string(25, 'F')));
        B = build_dedup({{dir + "/b1", dir + "/b2"}}, dir + "/e1", dir + "/e2", dp);
        CHECK(!B.ok, "a barcode read of the wrong length is refused when --soloBarcodeReadLength fixes it");
        DedupParams free_len = dp;
        free_len.barcode_len = 0;   // free length: STAR pads a short barcode read with N; here the UMI part is short, so N pads it
        B = build_dedup({{dir + "/b1", dir + "/b2"}}, dir + "/e1", dir + "/e2", free_len);
        CHECK(B.ok && B.table.n_ids() == 1 && MemberTable::unpack(B.table.umi[0], 10) == "AAAAACCCCN", "free barcode length: padded with N like STAR");
        put(dir + "/b2", fq("x", "AAACCCAAGAAACACTAAAAACCCC", std::string(25, 'F')) + fq("y", "AAACCCAAGAAACACTAAAAACCCC", std::string(25, 'F')));
        B = build_dedup({{dir + "/b1", dir + "/b2"}}, dir + "/e1", dir + "/e2", free_len);
        CHECK(!B.ok, "a barcode file with more records than the cDNA file is refused");
    }

    // ---- the junction files: today's form (every read, its own name) and the dedup form ----
    std::string today = kHdr, dedup = kHdr;
    for (size_t i = 0; i < all.size(); i++) today += lines_of(seq_of[idof[all[i].seq]], "r" + std::to_string(i), all[i].cb, all[i].umi);
    for (size_t id = 0; id < members.size(); id++) { const Rd& r = all[members[id][0]]; dedup += lines_of(seq_of[id], std::to_string(id), r.cb, r.umi); }
    today += "# 2.7.11b   STAR --chimSegmentMin 10\n";
    dedup += "# 2.7.11b   STAR --chimSegmentMin 10\n";
    put(dir + "/today.junction", today);
    put(dir + "/dedup.junction", dedup);
    const std::vector<std::string> bcs = {cbA, cbB, "AAACCCAAGAAACACG"};
    int cases = 0, equal = 0;
    for (int canonical_only = 0; canonical_only < 2; canonical_only++) {
        for (int rev = 0; rev < 2; rev++) {
            Config c0;
            c0.junction_path = dir + "/today.junction";
            c0.canonical_only = canonical_only == 1;
            c0.reverse_strand = rev == 1;
            Config c1 = c0;
            c1.junction_path = dir + "/dedup.junction";
            c1.members = &R.table;
            Stats s0, s1;
            const auto r0 = build_rows(bcs, c0, s0);
            const auto r1 = build_rows(bcs, c1, s1);
            cases++;
            const bool eq = s1.dedup_refusal.empty() && same_rows(r0, r1) && same_stats(s0, s1);
            equal += eq ? 1 : 0;
            CHECK(eq, "rows and every counter equal, canonical_only=" + std::to_string(canonical_only) + " reverse_strand=" + std::to_string(rev));
            if (canonical_only == 1 && rev == 0) {
                // 10 reads: sequence 0 has 6 reads x 4 lines, sequence 2 one line without barcode columns, sequence 3 one read x 1 line
                CHECK(s0.lines == 6 * 4 + 1 + 1 && s0.no_barcode == 1 && s0.ambiguous > 0 && s0.unmatched > 0 && s0.kept_lines > 0 && s0.noncanonical > 0 &&
                          s0.reoriented > 0,
                      "the synthetic file exercises every counter");
            }
        }
    }
    CHECK(cases == 4 && equal == 4, "four configurations equal");

    // ---- the expanded dump: every line once per member, name '.', the member's barcode ----
    {
        Config c1;
        c1.junction_path = dir + "/dedup.junction";
        c1.members = &R.table;
        c1.expanded_dump_path = dir + "/expanded.junction";
        Stats s1;
        build_rows(bcs, c1, s1);
        std::string want;
        for (size_t i = 0; i < all.size(); i++) want += lines_of(seq_of[idof[all[i].seq]], ".", all[i].cb, all[i].umi);
        std::vector<std::string> a, b;
        auto lines_of_text = [](const std::string& t) { std::vector<std::string> v; size_t p = 0; while (p < t.size()) { size_t q = t.find('\n', p); v.push_back(t.substr(p, q - p)); p = q + 1; } std::sort(v.begin(), v.end()); return v; };
        CHECK(lines_of_text(slurp(dir + "/expanded.junction")) == lines_of_text(want), "the dump is today's lines with the name replaced, as a multiset");
    }

    // ---- self-check: a printed barcode that is not the representative's ----
    {
        std::string bad = kHdr;
        bad += lines_of(0, "0", "AAACCCAAGAAACACT", "AAAAACCCCA");   // the representative's UMI is AAAAACCCCC
        put(dir + "/bad.junction", bad);
        Config c1;
        c1.junction_path = dir + "/bad.junction";
        c1.members = &R.table;
        c1.out_path = dir + "/bad.tsv.gz";
        Stats s1 = write_table(bcs, c1);
        CHECK(!s1.dedup_refusal.empty() && !s1.ok, "a printed UMI that differs from the member table: refused, no table written");
        std::string bad2 = kHdr;
        bad2 += lines_of(0, "99", cbA, "AAAAACCCCC");   // no such id
        put(dir + "/bad2.junction", bad2);
        c1.junction_path = dir + "/bad2.junction";
        Stats s2 = write_table(bcs, c1);
        CHECK(!s2.dedup_refusal.empty(), "an id outside the member table: refused");
    }

    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
