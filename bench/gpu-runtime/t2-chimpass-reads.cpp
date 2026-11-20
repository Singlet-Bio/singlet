// t2-chimpass-reads.cpp (chimpass task, throughput track 2): what the --chimeric second pass's input looks like, as counts only.
// Inputs: the pass's input FASTQ files of the cDNA mate (star_Unmapped.out.mate1 = reads the main pass left unmapped, star_Softclip.out.mate1 =
// mapped reads with >= 15 unaligned bases) and the pass's raw junction file (Chimeric.out.junction, kept with SINGLET_KEEP_CHIMERIC_JUNCTION=1).
// Output: "KEY value" lines on stdout. No read sequence, read name, barcode or UMI is printed.
// Build: g++ -O2 -fopenmp -std=c++17 -o t2-chimpass-reads t2-chimpass-reads.cpp
// Run:   t2-chimpass-reads --junction J --unm U1 --sc S1 --adapter AAGCAGTGGTATCAACGCAGAGTACATGGG --mmp 0.1 --minseg 10 --threads 16
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <omp.h>
#include <parallel/algorithm>
#include <string>
#include <vector>
typedef unsigned long long u64;
struct H128 {
    u64 a, b;
    bool operator<(const H128& o) const { return a < o.a || (a == o.a && b < o.b); }
    bool operator==(const H128& o) const { return a == o.a && b == o.b; }
};
static inline u64 mix(u64 x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static H128 hseq(const char* s, size_t n) {
    u64 a = 0xcbf29ce484222325ULL ^ n, b = 0x9e3779b97f4a7c15ULL + n;
    for (size_t i = 0; i < n; i++) {
        a = (a ^ (unsigned char)s[i]) * 0x100000001b3ULL;
        b = (b + (unsigned char)s[i] + 1) * 0xff51afd7ed558ccdULL;
        b ^= b >> 29;
    }
    H128 h; h.a = mix(a); h.b = mix(b); return h;
}
static u64 hname(const char* s, size_t n) { return hseq(s, n).a; }
static int code(char c) { switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; case 'T': return 3; default: return 4; } }

// the name as STAR keeps it: first token of the header without '@', without a trailing /1 or /2
static void nameOf(const char* h, size_t n, const char*& p, size_t& len) {
    p = h + 1; len = 0;
    while (1 + len < n && h[1 + len] != ' ' && h[1 + len] != '\t' && h[1 + len] != '\n' && h[1 + len] != '\r') len++;
    if (len >= 2 && p[len - 2] == '/' && (p[len - 1] == '1' || p[len - 1] == '2')) len -= 2;
}
struct Batch { std::vector<std::string> name, seq; };
// reads up to `max` FASTQ records; returns the number read
static size_t readBatch(FILE* f, Batch& b, size_t max) {
    b.name.clear(); b.seq.clear();
    char* line = nullptr; size_t cap = 0; ssize_t n;
    while (b.name.size() < max) {
        if ((n = getline(&line, &cap, f)) <= 0) break;
        std::string nm(line, n);
        if (getline(&line, &cap, f) <= 0) break;
        std::string sq(line);
        while (!sq.empty() && (sq.back() == '\n' || sq.back() == '\r')) sq.pop_back();
        if (getline(&line, &cap, f) <= 0) break;
        if (getline(&line, &cap, f) <= 0) break;
        b.name.push_back(std::move(nm)); b.seq.push_back(std::move(sq));
    }
    free(line);
    return b.name.size();
}

static const char* kBucketName[] = {"lt12", "12_19", "20_23", "24_29", "30_39", "40_49", "50_59", "60_69", "70_79", "80_89", "ge90"};
static int bucketOf(int L) { return L < 12 ? 0 : L < 20 ? 1 : L < 24 ? 2 : L < 30 ? 3 : L < 40 ? 4 : L < 50 ? 5 : L < 60 ? 6 : L < 70 ? 7 : L < 80 ? 8 : L < 90 ? 9 : 10; }
enum { C_N, C_SUMLC, C_CLIPPED, C_LC0, C_LT24, C_LT20, C_MONO_A, C_MONO_C, C_MONO_G, C_MONO_T, C_REP2_6, C_ENT10, C_ENT15, C_NANY, C_N10, C_TSO_M1, C_TSO_M1RC, C_TSO_M2, C_TSO_M2RC,
       C_TSO_ANY, C_TSO_2, C_B0, C_NC = C_B0 + 11 };
static const char* kCntName[] = {"reads", "sum_clipped_len", "clipped_by_adapter", "clipped_len_0", "clipped_len_lt_2x(minseg+2)", "clipped_len_lt_2xminseg", "mono_A_ge90pct", "mono_C_ge90pct", "mono_G_ge90pct", "mono_T_ge90pct",
    "tandem_period2_6_ge90pct_not_mono", "entropy_lt1.0_len_ge12", "entropy_lt1.5_len_ge12", "N_any", "N_ge10pct", "tso_5p_kit_TTTCTTATATGGG", "tso_5p_kit_revcomp", "tso_3p_kit_core16", "tso_3p_kit_core16_revcomp",
    "tso_any_motif", "tso_ge2_hits", "bucket_lt12", "bucket_12_19", "bucket_20_23", "bucket_24_29", "bucket_30_39", "bucket_40_49", "bucket_50_59", "bucket_60_69", "bucket_70_79", "bucket_80_89", "bucket_ge90"};
struct Cnt {
    u64 v[C_NC];
    Cnt() { memset(v, 0, sizeof v); }
    void add(const Cnt& o) { for (int i = 0; i < C_NC; i++) v[i] += o.v[i]; }
};

static int countHits(const std::string& s, const char* m, int ml) {   // non-overlapping occurrences of the exact motif
    int h = 0; int n = (int)s.size();
    for (int i = 0; i + ml <= n;) { if (memcmp(s.data() + i, m, ml) == 0) { h++; i += ml; } else i++; }
    return h;
}
static u64 distinctOf(const std::vector<H128>& v) { u64 d = 0; for (size_t i = 0; i < v.size(); i++) if (i == 0 || !(v[i] == v[i - 1])) d++; return d; }

int main(int argc, char** argv) {
    std::string J, U, S, adapter = "AAGCAGTGGTATCAACGCAGAGTACATGGG"; double mmp = 0.1; int minseg = 10; int threads = 16;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--junction") J = argv[i + 1]; else if (a == "--unm") U = argv[i + 1]; else if (a == "--sc") S = argv[i + 1];
        else if (a == "--adapter") adapter = argv[i + 1]; else if (a == "--mmp") mmp = atof(argv[i + 1]); else if (a == "--minseg") minseg = atoi(argv[i + 1]);
        else if (a == "--threads") threads = atoi(argv[i + 1]);
    }
    omp_set_num_threads(threads);
    std::vector<int> ad; for (char c : adapter) ad.push_back(code(c));
    const int minLen = 2 * (minseg + 2);   // two segments of >= minseg+2 bases (ChimericDetection_chimericDetectionMult.cpp: roE > segmentMin + roS + overlap)
    printf("INFO adapter_len %zu mmp %g minseg %d necessary_min_clipped_len %d threads %d\n", ad.size(), mmp, minseg, minLen, threads);

    // 1. names of the soft-clip route's reads
    std::vector<u64> scn;
    {
        FILE* f = fopen(S.c_str(), "r");
        if (!f) { printf("ERROR cannot open the soft-clip file\n"); return 1; }
        Batch b;
        while (readBatch(f, b, 1 << 21)) for (auto& h : b.name) { const char* p; size_t l; nameOf(h.data(), h.size(), p, l); scn.push_back(hname(p, l)); }
        fclose(f);
    }
    __gnu_parallel::sort(scn.begin(), scn.end());
    printf("SOFTCLIP_reads %zu\n", scn.size());
    auto isSc = [&](u64 h) { return std::binary_search(scn.begin(), scn.end(), h); };

    // 2. the junction file
    u64 lines = 0, bestLines[2] = {0, 0}, canonBest[2] = {0, 0}, noncanonBest[2] = {0, 0}, allLines[2] = {0, 0}, lineLt[2] = {0, 0};
    long minL = 1L << 30, minDrop = 1L << 30; u64 histAll[11] = {0}, histCanon[11] = {0};
    std::vector<u64> hAll, hKept;
    {
        FILE* f = fopen(J.c_str(), "r");
        if (!f) { printf("ERROR cannot open the junction file\n"); return 1; }
        char* line = nullptr; size_t cap = 0; ssize_t n;
        while ((n = getline(&line, &cap, f)) > 0) {
            if (line[0] == '#' || strncmp(line, "chr_donorA", 10) == 0) continue;
            const char* fld[32]; size_t fl[32]; int nf = 0; size_t st = 0;
            for (ssize_t i = 0; i <= n && nf < 32; i++)
                if (i == n || line[i] == '\t' || line[i] == '\n') { fld[nf] = line + st; fl[nf] = i - st; nf++; st = i + 1; if (i == n || line[i] == '\n') break; }
            if (nf < 19) continue;
            lines++;
            int jtype = atoi(std::string(fld[6], fl[6]).c_str());
            long L = atol(std::string(fld[15], fl[15]).c_str()), nc = atol(std::string(fld[16], fl[16]).c_str());
            long sc = atol(std::string(fld[17], fl[17]).c_str()), best = atol(std::string(fld[18], fl[18]).c_str());
            u64 h = hname(fld[9], fl[9]); int src = isSc(h) ? 1 : 0;
            allLines[src]++; minL = std::min(minL, L); minDrop = std::min(minDrop, L - nc); if (L < minLen) lineLt[src]++;
            histAll[bucketOf((int)L)]++; hAll.push_back(h);
            if (sc >= best) {
                bestLines[src]++;
                if (jtype >= 1) { canonBest[src]++; hKept.push_back(h); histCanon[bucketOf((int)L)]++; } else noncanonBest[src]++;
            }
        }
        free(line); fclose(f);
    }
    for (auto* v : {&hAll, &hKept}) { __gnu_parallel::sort(v->begin(), v->end()); v->erase(std::unique(v->begin(), v->end()), v->end()); }
    printf("JUNCTION_lines %llu\n", lines);
    for (int s = 0; s < 2; s++) {
        const char* sn = s ? "softclip" : "unmapped_or_other";
        printf("JUNCTION_lines_%s %llu\nJUNCTION_lines_%s_best_score %llu\nJUNCTION_lines_%s_best_canonical %llu\nJUNCTION_lines_%s_best_noncanonical %llu\nJUNCTION_lines_%s_with_maxposs_lt%d %llu\n",
               sn, allLines[s], sn, bestLines[s], sn, canonBest[s], sn, noncanonBest[s], sn, minLen, lineLt[s]);
    }
    printf("JUNCTION_min_maxposs_score %ld\nJUNCTION_min_drop_maxposs_minus_nonchim %ld\n", minL, minDrop);
    for (int i = 0; i < 11; i++) printf("JUNCTION_lines_by_clipped_len_%s all %llu best_canonical %llu\n", kBucketName[i], histAll[i], histCanon[i]);
    printf("JUNCTION_distinct_reads_any_line %zu\nJUNCTION_distinct_reads_best_canonical %zu\n", hAll.size(), hKept.size());
    {
        u64 a = 0, k = 0;
        for (u64 h : hAll) if (isSc(h)) a++;
        for (u64 h : hKept) if (isSc(h)) k++;
        printf("JUNCTION_distinct_reads_any_line_softclip %llu\nJUNCTION_distinct_reads_best_canonical_softclip %llu\n", a, k);
    }

    // 3. the reads
    static const char M1[] = "TTTCTTATATGGG", M1rc[] = "CCCATATAAGAAA", M2[] = "TATCAACGCAGAGTAC", M2rc[] = "GTACTCTGCGTTGATA";
    std::vector<H128> hs[2], hProdAny, hProdKept;
    static Cnt cnt[2][3];   // [source][0 all reads, 1 reads with a junction line, 2 reads with a best-score canonical line]
    u64 foundNames = 0;
    for (int src = 0; src < 2; src++) {
        FILE* f = fopen((src ? S : U).c_str(), "r");
        if (!f) { printf("ERROR cannot open an input file\n"); return 1; }
        Batch b;
        while (readBatch(f, b, 1 << 21)) {
            size_t nb = b.name.size();
            std::vector<H128> bh(nb); std::vector<unsigned char> pr(nb, 0);
            #pragma omp parallel
            {
                Cnt loc[3]; u64 fn = 0;
                #pragma omp for schedule(static)
                for (size_t r = 0; r < nb; r++) {
                    const std::string& s = b.seq[r]; const int n = (int)s.size();
                    bh[r] = hseq(s.data(), s.size());
                    const char* p; size_t l; nameOf(b.name[r].data(), b.name[r].size(), p, l);
                    const u64 hn = hname(p, l);
                    const bool any = std::binary_search(hAll.begin(), hAll.end(), hn), kept = any && std::binary_search(hKept.begin(), hKept.end(), hn);
                    pr[r] = (any ? 1 : 0) | (kept ? 2 : 0); if (any) fn++;
                    // 3' adapter clip as STAR's localSearch (ClipMate::clip, type 3p): the length that remains is ixBest
                    std::vector<int> x(n); for (int i = 0; i < n; i++) x[i] = code(s[i]);
                    int ixBest = n; long nMatchBest = 0, nMMbest = 0; const int ny = (int)ad.size();
                    for (int ix = 0; ix < n; ix++) {
                        const int m = std::min(ny, n - ix); if (m < nMatchBest) break;
                        long nMatch = 0, nMM = 0;
                        for (int iy = 0; iy < m; iy++) { if (x[ix + iy] > 3) continue; if (x[ix + iy] == ad[iy]) nMatch++; else nMM++; }
                        if ((nMatch > nMatchBest || (nMatch == nMatchBest && nMM < nMMbest)) && double(nMM) / double(nMatch) <= mmp) { ixBest = ix; nMatchBest = nMatch; nMMbest = nMM; }
                    }
                    const int Lc = ixBest;
                    int cb[5] = {0, 0, 0, 0, 0}; for (int i = 0; i < Lc; i++) cb[x[i]]++;
                    int nn = cb[4]; const int nacgt = Lc - nn;
                    double ent = 0; for (int k = 0; k < 4; k++) if (cb[k] > 0) { double q = double(cb[k]) / std::max(1, nacgt); ent -= q * log2(q); }
                    int mono = -1; for (int k = 0; k < 4; k++) if (Lc > 0 && cb[k] * 10 >= Lc * 9) mono = k;
                    bool rep = false;
                    if (mono < 0 && Lc >= 12)
                        for (int per = 2; per <= 6 && !rep; per++) { int same = 0; for (int i = per; i < Lc; i++) if (x[i] == x[i - per] && x[i] < 4) same++; if (same * 10 >= (Lc - per) * 9) rep = true; }
                    const int h1 = countHits(s, M1, 13), h2 = countHits(s, M1rc, 13), h3 = countHits(s, M2, 16), h4 = countHits(s, M2rc, 16);
                    const int hits = h1 + h2 + h3 + h4;
                    for (int c = 0; c < 3; c++) {
                        if (c == 1 && !any) continue;
                        if (c == 2 && !kept) continue;
                        Cnt& k = loc[c];
                        k.v[C_N]++; k.v[C_SUMLC] += Lc; if (Lc < n) k.v[C_CLIPPED]++; if (Lc == 0) k.v[C_LC0]++; if (Lc < minLen) k.v[C_LT24]++; if (Lc < 2 * minseg) k.v[C_LT20]++;
                        if (mono >= 0) k.v[C_MONO_A + mono]++;
                        if (rep) k.v[C_REP2_6]++;
                        if (Lc >= 12 && ent < 1.0) k.v[C_ENT10]++;
                        if (Lc >= 12 && ent < 1.5) k.v[C_ENT15]++;
                        if (nn > 0) k.v[C_NANY]++;
                        if (Lc > 0 && nn * 10 >= Lc) k.v[C_N10]++;
                        if (h1) k.v[C_TSO_M1]++; if (h2) k.v[C_TSO_M1RC]++; if (h3) k.v[C_TSO_M2]++; if (h4) k.v[C_TSO_M2RC]++;
                        if (hits >= 1) k.v[C_TSO_ANY]++;
                        if (hits >= 2) k.v[C_TSO_2]++;
                        k.v[C_B0 + bucketOf(Lc)]++;
                    }
                }
                #pragma omp critical
                { for (int c = 0; c < 3; c++) cnt[src][c].add(loc[c]); foundNames += fn; }
            }
            hs[src].insert(hs[src].end(), bh.begin(), bh.end());
            for (size_t r = 0; r < nb; r++) { if (pr[r] & 1) hProdAny.push_back(bh[r]); if (pr[r] & 2) hProdKept.push_back(bh[r]); }
        }
        fclose(f);
    }
    printf("JUNCTION_reads_found_in_the_two_input_files %llu\n", foundNames);
    const char* cn[3] = {"all", "with_a_junction_line", "with_a_best_canonical_line"};
    for (int s = 0; s < 2; s++) for (int c = 0; c < 3; c++) for (int k = 0; k < C_NC; k++)
        printf("CLASS source=%s set=%s %s %llu\n", s ? "softclip" : "unmapped", cn[c], kCntName[k], cnt[s][c].v[k]);

    // 4. distinct sequences
    for (int s = 0; s < 2; s++) __gnu_parallel::sort(hs[s].begin(), hs[s].end());
    printf("DISTINCT unmapped reads %zu distinct %llu\nDISTINCT softclip reads %zu distinct %llu\n", hs[0].size(), distinctOf(hs[0]), hs[1].size(), distinctOf(hs[1]));
    std::vector<H128> all; all.reserve(hs[0].size() + hs[1].size());
    all.insert(all.end(), hs[0].begin(), hs[0].end()); all.insert(all.end(), hs[1].begin(), hs[1].end());
    std::vector<H128>().swap(hs[0]); std::vector<H128>().swap(hs[1]);
    __gnu_parallel::sort(all.begin(), all.end());
    printf("DISTINCT all reads %zu distinct %llu\n", all.size(), distinctOf(all));
    {
        static const u64 edge[] = {1, 2, 5, 10, 100, 1000, 10000, ~0ULL};
        static const char* en[] = {"1", "2", "3_5", "6_10", "11_100", "101_1000", "1001_10000", "gt10000"};
        u64 nSeq[8] = {0}, nRead[8] = {0}; std::vector<u64> top; size_t i = 0;
        while (i < all.size()) {
            size_t j = i; while (j < all.size() && all[j] == all[i]) j++;
            u64 m = j - i; int bk = 0; while (m > edge[bk]) bk++;
            nSeq[bk]++; nRead[bk] += m; top.push_back(m); i = j;
        }
        for (int bk = 0; bk < 8; bk++) printf("MULTIPLICITY %s distinct_sequences %llu reads %llu\n", en[bk], nSeq[bk], nRead[bk]);
        size_t nt = std::min<size_t>(10, top.size());
        std::partial_sort(top.begin(), top.begin() + nt, top.end(), std::greater<u64>());
        for (size_t k = 0; k < nt; k++) printf("MULTIPLICITY top%zu_count %llu\n", k + 1, top[k]);
    }
    for (auto* v : {&hProdAny, &hProdKept}) __gnu_parallel::sort(v->begin(), v->end());
    printf("DISTINCT reads_with_a_junction_line %zu distinct %llu\nDISTINCT reads_with_a_best_canonical_line %zu distinct %llu\n", hProdAny.size(), distinctOf(hProdAny), hProdKept.size(), distinctOf(hProdKept));
    {   // does one sequence always give the same verdict? (a sequence is mixed when it has more reads in `all` than reads with a best canonical line)
        const std::vector<H128>& pk = hProdKept; u64 seqKept = 0, seqMixed = 0; size_t a = 0, p = 0;
        while (p < pk.size()) {
            size_t pj = p; while (pj < pk.size() && pk[pj] == pk[p]) pj++;
            while (a < all.size() && all[a] < pk[p]) a++;
            size_t aj = a; while (aj < all.size() && all[aj] == pk[p]) aj++;
            seqKept++; if (aj - a > pj - p) seqMixed++;
            a = aj; p = pj;
        }
        printf("DISTINCT sequences_with_a_best_canonical_line_in_some_read %llu of_which_other_reads_of_the_same_sequence_have_none %llu\n", seqKept, seqMixed);
    }
    return 0;
}
