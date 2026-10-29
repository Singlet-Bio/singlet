// Shared, CUDA-free part of the phase 1d harness: parameters, STAR genome loading, the unit-case fixtures, and the host run of one
// case. Included by paired_microbench.cu (GPU) and paired_unit_host.cpp (g++ only: runs the kernel stage on the host against the
// Python model's results, which is the check that needs no GPU). DRAFT: the GPU parts are not compiled.
#pragma once
#include "paired_windows.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using m5::u64;
using m5::i64;
using m5::Params;
using m5::GenomeView;
using m5::pe::PeLane;
using m5::pe::WorkPE;
typedef m5::Work<PeLane>::T Tr;
typedef m5::ReadBuf<PeLane::LREAD> RBuf;

// ---------------------------------------------------------------------------------------------------------------------
// Parameters: STAR's defaults (parametersDefault) for the fields the kernel reads, and Genome_genomeLoad.cpp:419-449
// ---------------------------------------------------------------------------------------------------------------------
static int glScore(double scale, u64 x) { return int(std::ceil(std::log2((double)x) * scale - 0.5)); }

static Params defaultParams() {
    Params Q{};
    Q.maxNsplit = 10; Q.seedSplitMin = 12; Q.seedSearchStartLmax = 50; Q.seedSearchStartLmaxOverLread = 1.0; Q.seedMapMin = 5;
    Q.seedSearchLmax = 0; Q.seedMultimapNmax = 10000; Q.seedPerReadNmax = 1000; Q.seedPerWindowNmax = 50;
    Q.winBinNbits = 16; Q.winAnchorDistNbins = 9; Q.winFlankNbins = 4; Q.winAnchorMultimapNmax = 50;
    Q.alignWindowsPerReadNmax = 10000; Q.alignTranscriptsPerWindowNmax = 100; Q.alignTranscriptsPerReadNmax = 10000;
    Q.alignSJoverhangMin = 5; Q.alignSJDBoverhangMin = 3; Q.alignSplicedMateMapLmin = 0; Q.alignSplicedMateMapLminOverLmate = 0.66;
    Q.alignIntronMin = 21; Q.alignIntronMax = 0; Q.alignMatesGapMax = 0;
    Q.alignSJstitchMismatchNmax[0] = 0; Q.alignSJstitchMismatchNmax[1] = -1; Q.alignSJstitchMismatchNmax[2] = 0; Q.alignSJstitchMismatchNmax[3] = 0;
    Q.alignSoftClipAtReferenceEnds = true;
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) Q.alignEndsTypeExt[a][b] = false;   // Local
    Q.flushRight = false; Q.alignEndsProtrudeNbasesMax = 0;
    Q.intronMotifsFilter = 0; Q.removeInconsistentStrands = true; Q.outSAMstrandFieldType = 0;
    Q.scoreGenomicLengthLog2scale = -0.25;
    Q.outFilterMultimapScoreRange = 1; Q.outFilterMultimapNmax = 10; Q.multOrderRandom = false; Q.outSAMmultNmax = (u64)-1; Q.primaryAllBest = false;
    Q.outFilterScoreMin = 0; Q.outFilterScoreMinOverLread = 0.66; Q.outFilterMatchNmin = 0; Q.outFilterMatchNminOverLread = 0.66;
    Q.outFilterMismatchNmax = 10; Q.outFilterMismatchNoverLmax = 0.3; Q.outFilterMismatchNoverReadLmax = 1.0;
    Q.scoreGap = 0; Q.scoreGapNoncan = -8; Q.scoreGapGCAG = -4; Q.scoreGapATAC = -8; Q.scoreDelBase = -2; Q.scoreDelOpen = -2;
    Q.scoreInsBase = -2; Q.scoreInsOpen = -2; Q.scoreStitchSJshift = 1; Q.sjdbScore = 2;
    Q.gSAsparseD = 1; Q.gSAindexNbases = 14;
    return Q;
}

static bool setParam(Params& Q, const std::string& k, double v) {
    if (k == "alignEndsProtrudeNbasesMax") Q.alignEndsProtrudeNbasesMax = (int)v;
    else if (k == "alignIntronMax") Q.alignIntronMax = (u64)v;
    else if (k == "alignMatesGapMax") Q.alignMatesGapMax = (u64)v;
    else if (k == "outFilterMultimapNmax") Q.outFilterMultimapNmax = (u64)v;
    else if (k == "outFilterMismatchNmax") Q.outFilterMismatchNmax = (u64)v;
    else if (k == "alignSplicedMateMapLminOverLmate") Q.alignSplicedMateMapLminOverLmate = v;
    else return false;
    return true;
}

// the genome-dependent redefinitions (Genome_genomeLoad.cpp:419-449) and the genomic length breakpoints (SeqCacheGpuPrefill.cpp fillParams)
static bool finishParams(Params& Q, u64 nGenome, u64 gChrBinNbits, std::string& why) {
    if (!(Q.alignIntronMax == 0 && Q.alignMatesGapMax == 0)) {
        u64 gap = Q.alignMatesGapMax == 0 ? 1000ULL : Q.alignMatesGapMax;
        Q.winBinNbits = (u64)std::floor(std::log2((double)(std::max(std::max(4ULL, Q.alignIntronMax), gap) / 4)) + 0.5);
        Q.winBinNbits = std::max(Q.winBinNbits, (u64)std::floor(std::log2((double)(nGenome / 40000 + 1)) + 0.5));
    }
    if (Q.winBinNbits > gChrBinNbits) Q.winBinNbits = gChrBinNbits;
    if (!(Q.alignIntronMax == 0 && Q.alignMatesGapMax == 0)) {
        Q.winFlankNbins = std::max(Q.alignIntronMax, Q.alignMatesGapMax) / (1ULL << Q.winBinNbits) + 1;
        Q.winAnchorDistNbins = 2 * Q.winFlankNbins;
    }
    Q.winBinChrNbits = gChrBinNbits - Q.winBinNbits;
    Q.winBinN = nGenome / (1ULL << Q.winBinNbits) + 1;
    double s = Q.scoreGenomicLengthLog2scale;
    Q.glVal0 = glScore(s, 1); Q.glN = 0;
    u64 x = 1; int v = Q.glVal0; const u64 XMAX = 1ULL << 40;
    while (x < XMAX && s != 0) {
        u64 lo = x, hi = x + 1;
        while (hi < XMAX && glScore(s, hi) == v) { lo = hi; hi = std::min(XMAX, hi * 2); }
        if (glScore(s, hi) == v) break;
        while (hi - lo > 1) { u64 m = lo + (hi - lo) / 2; if (glScore(s, m) == v) lo = m; else hi = m; }
        if (Q.glN >= 64) { why = "genomic length score breakpoints"; return false; }
        v = glScore(s, hi); Q.glBreak[Q.glN] = hi; Q.glVal[Q.glN] = v; Q.glN++; x = hi;
    }
    for (u64 xx = 1; xx < (1ULL << 22); xx += 1 + xx / 997)
        if (m5::genomicLengthScore(Q, xx) != glScore(s, xx)) { why = "genomic length score check"; return false; }
    return true;
}

static void applyKind(Params& Q, const std::string& kind) {
    if (kind == "atac") { Q.alignIntronMax = 1; Q.alignMatesGapMax = 2000; }   // --alignIntronMax 1 --alignMatesGapMax 2000
}

// ---------------------------------------------------------------------------------------------------------------------
// Genome: host arrays and the view. G has STAR's 200-byte pads of 5 on both sides.
// ---------------------------------------------------------------------------------------------------------------------
struct HostGenome {
    std::vector<char> Gbuf;
    std::vector<u64> SA, SAi, chrStart, chrLength, chrBin;
    GenomeView v{};
    u64 gChrBinNbits = 18;
    std::vector<std::string> chrName;
    void finishView() {
        v.G = Gbuf.data() + 200;
        v.SA = SA.empty() ? nullptr : SA.data();
        v.SAi = SAi.empty() ? nullptr : SAi.data();
        v.chrStart = chrStart.data(); v.chrLength = chrLength.data(); v.chrBin = chrBin.data();
        v.nChr = chrLength.size(); v.chrBinN = chrBin.size();
        v.sjGstart = chrStart[v.nChr] + 1;   // no sjdb: no coordinate reaches it (Genome_genomeLoad.cpp:515)
        v.sjdbOverhang = 0; v.sjdbLength = 0; v.sjdbN = 0;
        v.sjdbStart = v.sjdbEnd = v.sjDstart = v.sjAstart = nullptr;
        v.sjdbMotif = v.sjdbShiftLeft = v.sjdbShiftRight = v.sjdbStrand = nullptr;
    }
    void fillChrBin() {   // Genome::chrBinFill (Genome.cpp:209)
        u64 n = chrStart[chrLength.size()] / (1ULL << gChrBinNbits) + 1;
        chrBin.assign(n, 0);
        for (u64 ii = 0, ichr = 1; ii < n; ++ii) {
            if (ichr < chrStart.size() && ii * (1ULL << gChrBinNbits) >= chrStart[ichr]) ichr++;
            chrBin[ii] = ichr - 1;
        }
    }
};

static std::vector<char> slurp(const std::string& f) {
    std::ifstream in(f, std::ios::binary | std::ios::ate);
    if (!in) { fprintf(stderr, "cannot open %s\n", f.c_str()); exit(1); }
    std::vector<char> b((size_t)in.tellg());
    in.seekg(0);
    in.read(b.data(), (std::streamsize)b.size());
    return b;
}
static std::vector<u64> readNumbers(const std::string& f) {
    std::ifstream in(f);
    if (!in) { fprintf(stderr, "cannot open %s\n", f.c_str()); exit(1); }
    std::vector<u64> v; u64 x;
    while (in >> x) v.push_back(x);
    return v;
}

// STAR genome directory (Genome_genomeLoad.cpp:34-188): Genome, SA, SAindex, chrStart.txt, chrLength.txt, genomeParameters.txt
static bool loadStarGenome(const std::string& dir, HostGenome& H, Params& Q, std::string& why) {
    if (std::ifstream(dir + "/sjdbInfo.txt").good()) { why = "the genome has junctions (sjdbInfo.txt): build it without --sjdbGTFfile"; return false; }
    std::map<std::string, std::string> par;
    u64 gsb = 0;
    {
        std::ifstream in(dir + "/genomeParameters.txt");
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string k, v; ls >> k >> v;
            if (k == "###") { std::string k2, v2; ls >> v2; k2 = v; if (k2 == "GstrandBit") gsb = std::strtoull(v2.c_str(), nullptr, 10); continue; }
            if (!k.empty()) par[k] = v;
        }
    }
    H.gChrBinNbits = std::strtoull(par["genomeChrBinNbits"].c_str(), nullptr, 10);
    Q.gSAindexNbases = std::strtoull(par["genomeSAindexNbases"].c_str(), nullptr, 10);
    Q.gSAsparseD = std::max<u64>(1, std::strtoull(par["genomeSAsparseD"].c_str(), nullptr, 10));
    if (H.gChrBinNbits == 0 || Q.gSAindexNbases == 0) { why = "genomeParameters.txt: no genomeChrBinNbits / genomeSAindexNbases"; return false; }
    if (Q.gSAsparseD > 4) { why = "genomeSAsparseD > 4"; return false; }
    H.chrStart = readNumbers(dir + "/chrStart.txt");
    H.chrLength = readNumbers(dir + "/chrLength.txt");
    std::vector<char> g = slurp(dir + "/Genome");
    u64 nGenome = g.size();
    H.Gbuf.assign(nGenome + 400, 5);
    memcpy(H.Gbuf.data() + 200, g.data(), nGenome);
    if (gsb == 0) { gsb = (u64)std::floor(std::log2((double)nGenome)) + 1; if (gsb < 32) gsb = 32; }
    std::vector<char> sa = slurp(dir + "/SA");
    H.SA.assign((sa.size() + 7) / 8 + 2, 0); memcpy(H.SA.data(), sa.data(), sa.size());
    std::vector<char> sai = slurp(dir + "/SAindex");
    u64 L = Q.gSAindexNbases;
    std::vector<u64> start(L + 1);
    memcpy(start.data(), sai.data() + 8, 8 * (L + 1));
    size_t hdr = 8 + 8 * (L + 1);
    H.SAi.assign((sai.size() - hdr + 7) / 8 + 2, 0); memcpy(H.SAi.data(), sai.data() + hdr, sai.size() - hdr);
    GenomeView& v = H.v;
    v.nGenome = nGenome; v.GstrandBit = gsb; v.GstrandMask = ~(1ULL << gsb);
    v.saBits = gsb + 1; v.nSA = (sa.size() * 8) / v.saBits; v.saiBits = gsb + 3;
    v.SAiMarkNmaskC = 1ULL << (gsb + 1); v.SAiMarkNmask = ~v.SAiMarkNmaskC; v.SAiMarkAbsentMaskC = 1ULL << (gsb + 2);
    for (u64 i = 0; i <= L && i < 32; i++) v.saiStart[i] = start[i];
    H.fillChrBin();
    H.finishView();
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Unit mode
// ---------------------------------------------------------------------------------------------------------------------
struct ExTr { int nExons; i64 score, nMatch, nMM, gLength, iFrag, Str, Chr, primary; std::vector<std::array<i64, 4>> ex; std::vector<i64> sj; };
struct Case {
    std::string name, kind; std::vector<std::pair<std::string, double>> ov;
    std::string m1, m2;
    struct Win { u64 Str, Chr; i64 lastAnchor; std::vector<std::array<i64, m5::WA_SIZE>> WA; };
    std::vector<Win> wins;
    int unmapType = 0, nW = 0, primary = -1; std::vector<ExTr> trs;
};

static char nuc(char c) { switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; case 'T': return 3; default: return 4; } }

struct ClipCase { std::string name, adapter, seq; uint32_t n5, after5, n3, after3; double mmp; u64 len, start, c5, c3; };

static bool parseCases(const std::string& file, std::map<std::string, std::string>& chroms, std::vector<std::string>& order, u64& gbits, std::vector<Case>& cases, std::vector<ClipCase>& clips) {
    std::ifstream in(file);
    if (!in) return false;
    std::string line; Case* c = nullptr;
    while (std::getline(in, line)) {
        std::istringstream ls(line); std::string tag; ls >> tag;
        if (tag == "GENOME") { int n; ls >> gbits >> n; }
        else if (tag == "CHR") { std::string n, s; ls >> n >> s; chroms[n] = s; order.push_back(n); }
        else if (tag == "CASE") { cases.emplace_back(); c = &cases.back(); int k; ls >> c->name >> c->kind >> k; for (int i = 0; i < k; i++) { std::string key; double v; ls >> key >> v; c->ov.push_back({key, v}); } }
        else if (tag == "CLIPCASE") { ClipCase k; ls >> k.name >> k.n5 >> k.after5 >> k.n3 >> k.after3 >> k.mmp >> k.adapter >> k.seq >> k.len >> k.start >> k.c5 >> k.c3; clips.push_back(k); }
        else if (tag == "READ") ls >> c->m1 >> c->m2;
        else if (tag == "NWIN") { c->wins.clear(); }   // the WIN lines follow, one per window (empty windows included)
        else if (tag == "WIN") {
            Case::Win w; int nWA; ls >> w.Str >> w.Chr >> w.lastAnchor >> nWA; w.WA.resize(nWA);
            for (auto& a : w.WA) for (int k = 0; k < m5::WA_SIZE; k++) ls >> a[k];
            c->wins.push_back(w);
        }
        else if (tag == "EXPECT") { int n; ls >> c->unmapType >> c->nW >> c->primary >> n; }
        else if (tag == "TR") {
            ExTr t; ls >> t.nExons >> t.score >> t.nMatch >> t.nMM >> t.gLength >> t.iFrag >> t.Str >> t.Chr >> t.primary;
            t.ex.resize(t.nExons); for (auto& e : t.ex) ls >> e[0] >> e[1] >> e[2] >> e[3];
            t.sj.resize(t.nExons - 1); for (auto& s : t.sj) ls >> s;
            c->trs.push_back(t);
        }
    }
    return true;
}

// the state mapPhase1 + assignAll leave (stitch.cuh:1346), rebuilt from the model's windows; rb gets the three read buffers
static void setupWork(WorkPE<PeLane>& w, RBuf& rb, const Params& P, const Case& c) {
    memset(&w, 0, sizeof(w));
    std::vector<char> m1(c.m1.size()), m2(c.m2.size()), comb(c.m1.size() + c.m2.size() + 1);
    for (size_t i = 0; i < m1.size(); i++) m1[i] = nuc(c.m1[i]);
    for (size_t i = 0; i < m2.size(); i++) m2[i] = nuc(c.m2[i]);
    uint32_t L0 = (uint32_t)m1.size(), L1 = (uint32_t)m2.size();
    u64 Lread = m5::pe::combineMates(m1.data(), L0, m2.data(), L1, comb.data());
    for (u64 i = 0; i < Lread; i++) { char x = comb[i]; rb.r[0][i] = x; rb.r[1][i] = x < 4 ? 3 - x : x; }
    for (u64 i = 0; i < Lread; i++) rb.r[2][Lread - i - 1] = rb.r[1][i];
    w.Lread = Lread; w.readLength[0] = L0; w.readLength[1] = L1;
    w.outFilterMismatchNmaxTotal = std::min<u64>(P.outFilterMismatchNmax, (u64)(P.outFilterMismatchNoverReadLmax * (L0 + L1)));
    w.stitched = !c.wins.empty(); w.nW = c.wins.size();
    for (size_t i = 0; i < c.wins.size(); i++) {
        w.WC[i][m5::WC_Str] = c.wins[i].Str; w.WC[i][m5::WC_Chr] = c.wins[i].Chr;
        w.nWA[i] = c.wins[i].WA.size(); w.WlastAnchor[i] = c.wins[i].lastAnchor;
        for (size_t a = 0; a < c.wins[i].WA.size(); a++) for (int k = 0; k < m5::WA_SIZE; k++) w.WA[i][a][k] = (u64)c.wins[i].WA[a][k];
    }
    w.trBestKeep = -1; w.trBestScore = 0; w.trBestGLength = 0; w.nKeep = 0; w.nWtr = 0; w.trNtotal = 0; w.ov = m5::OV_NONE;
}

static std::string describe(const WorkPE<PeLane>& w) {
    std::ostringstream o;
    o << "unmapType=" << w.unmapType << " nW=" << w.nW << " nTr=" << w.nTr;
    for (u64 i = 0; i < w.nTr; i++) {
        const Tr& t = w.keep[i];
        o << "\n    tr" << i << " score=" << t.maxScore << " iFrag=" << t.iFrag << " primary=" << t.primaryFlag << " exons=";
        for (u64 e = 0; e < t.nExons; e++) o << "[" << t.exons[e][m5::EX_R] << "," << t.exons[e][m5::EX_G] << "," << t.exons[e][m5::EX_L] << "," << t.exons[e][m5::EX_iFrag] << "]";
    }
    return o.str();
}

static bool matches(const WorkPE<PeLane>& w, const Case& c) {
    if (w.unmapType != c.unmapType || (int)w.nW != c.nW || w.nTr != c.trs.size()) return false;
    int prim = -1;
    for (u64 i = 0; i < w.nTr; i++) {
        const Tr& t = w.keep[i]; const ExTr& e = c.trs[i];
        if (t.primaryFlag) prim = (int)i;
        if ((int)t.nExons != e.nExons || t.maxScore != e.score || (i64)t.nMatch != e.nMatch || (i64)t.nMM != e.nMM || (i64)t.gLength != e.gLength
            || t.iFrag != e.iFrag || (i64)t.Str != e.Str || (i64)t.Chr != e.Chr) return false;
        for (int x = 0; x < e.nExons; x++)
            if ((i64)t.exons[x][m5::EX_R] != e.ex[x][0] || (i64)t.exons[x][m5::EX_G] != e.ex[x][1] || (i64)t.exons[x][m5::EX_L] != e.ex[x][2]
                || (i64)t.exons[x][m5::EX_iFrag] != e.ex[x][3]) return false;
        for (int x = 0; x + 1 < e.nExons; x++) if ((i64)t.canonSJ[x] != e.sj[x]) return false;
    }
    return prim == c.primary;
}


// the unit genome as the model's Genome builds it: chromosomes at multiples of 2^gbits, padding 5
static void buildUnitGenome(HostGenome& H, const std::map<std::string, std::string>& chroms, const std::vector<std::string>& order, u64 gbits) {
    H.gChrBinNbits = gbits;
    u64 step = 1ULL << gbits, pos = 0;
    for (auto& n : order) { H.chrStart.push_back(pos); H.chrLength.push_back(chroms.at(n).size()); pos += ((chroms.at(n).size() + step) / step) * step; }
    H.chrStart.push_back(pos);
    H.Gbuf.assign(pos + 400, 5);
    for (size_t i = 0; i < order.size(); i++) for (size_t j = 0; j < chroms.at(order[i]).size(); j++) H.Gbuf[200 + H.chrStart[i] + j] = nuc(chroms.at(order[i])[j]);
    H.v.nGenome = pos;
    H.fillChrBin(); H.finishView();
}

// parameters of one case, as the model's cases.params builds them
static bool caseParams(const Case& c, u64 nGenome, u64 gbits, Params& P) {
    P = defaultParams(); applyKind(P, c.kind);
    for (auto& kv : c.ov) if (!setParam(P, kv.first, kv.second)) { fprintf(stderr, "unknown override %s\n", kv.first.c_str()); return false; }
    std::string why;
    if (!finishParams(P, nGenome, gbits, why)) { fprintf(stderr, "params: %s\n", why.c_str()); return false; }
    return true;
}

// the clipping unit check: m5::pe::clipMate against the model's clip_mate. Returns the number of differences.
static int runClipCases(const std::vector<ClipCase>& clips) {
    int bad = 0;
    for (auto& k : clips) {
        std::vector<char> seq(k.seq.size()), ad;
        for (size_t i = 0; i < seq.size(); i++) seq[i] = nuc(k.seq[i]);
        if (k.adapter == "polyA") ad.assign(m5::DEF_readSeqLengthMax, 0);
        else if (k.adapter != "-") for (char ch : k.adapter) ad.push_back(nuc(ch));
        m5::pe::MateClip c{k.n5, k.after5, k.n3, k.after3, (uint32_t)ad.size(), k.mmp, ad.data()};
        m5::pe::ClipSpan r = m5::pe::clipMate(c, seq.data(), seq.size());
        bool ok = r.len == k.len && (r.len == 0 || r.start == k.start) && r.c5 == k.c5 && r.c3 == k.c3;
        printf("clip %-28s %s\n", k.name.c_str(), ok ? "ok" : "DIFF");
        if (!ok) { printf("  kernel len %llu start %llu c5 %llu c3 %llu; model len %llu start %llu c5 %llu c3 %llu\n", (unsigned long long)r.len, (unsigned long long)r.start,
                          (unsigned long long)r.c5, (unsigned long long)r.c3, (unsigned long long)k.len, (unsigned long long)k.start, (unsigned long long)k.c5, (unsigned long long)k.c3); bad++; }
    }
    return bad;
}
