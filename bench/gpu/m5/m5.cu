// M5: STAR's mapping after the seed search, on the GPU, checked byte for byte against STAR (stardump M5 truth).
// docs/gpu-engine/exploration.md §12.13. The port itself: stitch.cuh.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -Xcompiler -fopenmp -o m5 m5.cu
// Run:   m5 --genome-dir STAR_INDEX --truth PREFIX [--reps 3] [--json OUT] [--blocks-per-sm 4] [--threads 128]
//          PREFIX: stardump's STARDUMP_M5 prefix (PREFIX.<tid>.bin, PREFIX.params.txt, PREFIX.genome.bin; m5_dump.h)
//
// Every truth read (a distinct cDNA sequence after clipping, as STAR mapped it) runs through mapRead<Main> on the GPU,
// one read per thread. Reads that exceed a main-lane capacity run through mapRead<Heavy> (STAR's own limits) on the host
// with the same source. The result (the sequence cache's encoding of STAR's mapping state, plus unmapType) is compared
// with STAR's: identical or not, per read and weighted by the read multiplicity.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <glob.h>
#include <sys/stat.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "stitch.cuh"

using namespace m5;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

//                LREAD NSPLIT  NP   NW  NWA  NWB  NEX NWTR NKEEP NBLOB
typedef Caps<     160,     4,  32,  16,  16, 256,   8,   8,    8,  640> Main;
typedef Caps<     650,    50, 1000, 10000, 50, 1 << 20, 20, 100, 4096, 1 << 20> Heavy;   // STAR: seedPerReadNmax, alignWindowsPerReadNmax, seedPerWindowNmax, MAX_N_EXONS, alignTranscriptsPerWindowNmax

// ---------------------------------------------------------------------------------------------------------------------
__global__ void k_map(Params P, GenomeView g, const char* rb, const u64* roff, const uint16_t* rlen, u64 n,
                      Work<Main>* work, ReadBuf<Main::LREAD>* rbuf, uint8_t* out, uint16_t* outLen, uint8_t* status, uint8_t* why,
                      int8_t* unmap) {
    u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, nt = (u64)gridDim.x * blockDim.x;
    Work<Main>& w = work[tid];
    ReadBuf<Main::LREAD>& b = rbuf[tid];
    for (u64 i = tid; i < n; i += nt) {
        Blob bl{out + i * Main::NBLOB, 0, Main::NBLOB, false};
        Status s = mapRead(P, g, w, b, rb + roff[i], rlen[i], bl);
        status[i] = (uint8_t)s; why[i] = (uint8_t)w.ov; outLen[i] = (uint16_t)bl.n; unmap[i] = (int8_t)w.unmapType;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
static u64 fileSize(const std::string& p) { struct stat st; if (stat(p.c_str(), &st)) { perror(p.c_str()); exit(2); } return st.st_size; }
static void readInto(const std::string& path, u64 off, char* dst, u64 n) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { perror(path.c_str()); exit(2); }
    fseeko(f, (off_t)off, SEEK_SET);
    u64 got = 0; while (got < n) { size_t k = fread(dst + got, 1, std::min<u64>(n - got, 1ULL << 30), f); if (!k) break; got += k; }
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
}
template <class T> static T* toDev(const T* h, u64 n) {
    T* d; CK(cudaMalloc(&d, std::max<u64>(1, n) * sizeof(T)));
    if (n) CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static T* toDev(const std::vector<T>& v) { return toDev(v.data(), (u64)v.size()); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct ParamFile {
    std::map<std::string, std::string> kv;
    void load(const std::string& p) {
        FILE* f = fopen(p.c_str(), "r"); if (!f) { perror(p.c_str()); exit(2); }
        char line[8192];
        while (fgets(line, sizeof line, f)) {
            std::string s(line); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            size_t sp = s.find(' '); if (sp == std::string::npos) continue;
            kv[s.substr(0, sp)] = s.substr(sp + 1);
        }
        fclose(f);
    }
    const std::string& str(const char* k) { auto it = kv.find(k); if (it == kv.end()) { fprintf(stderr, "param %s missing\n", k); exit(2); } return it->second; }
    u64 u(const char* k) { return strtoull(str(k).c_str(), nullptr, 10); }
    long long i(const char* k) { return strtoll(str(k).c_str(), nullptr, 10); }
    double d(const char* k) { return strtod(str(k).c_str(), nullptr); }
};

static std::vector<char> loadFile(const std::string& p) { std::vector<char> b(fileSize(p)); readInto(p, 0, b.data(), b.size()); return b; }
template <class T> static std::vector<T> takeArr(const std::vector<char>& b, u64& o) {
    u64 n; memcpy(&n, b.data() + o, 8); o += 8;
    std::vector<T> v(n); if (n) memcpy(v.data(), b.data() + o, n * sizeof(T)); o += n * sizeof(T);
    return v;
}

struct Truth {
    std::vector<char> rb; std::vector<u64> roff{0}; std::vector<uint16_t> rlen; std::vector<uint32_t> count;
    std::vector<uint8_t> blob; std::vector<u64> boff{0}; std::vector<int8_t> unmap; std::vector<uint8_t> flags; std::vector<uint32_t> marker;
};
#pragma pack(push, 1)
struct M5Rec { uint64_t iReadAll; uint32_t count; uint16_t Lread; uint8_t flags, pad; int32_t unmapType; uint32_t mapMarker, blobLen; };
#pragma pack(pop)
static void loadTruth(const std::string& prefix, Truth& T) {
    glob_t g; std::string pat = prefix + ".*.bin";
    if (glob(pat.c_str(), 0, nullptr, &g)) { fprintf(stderr, "no truth files %s\n", pat.c_str()); exit(2); }
    for (size_t fi = 0; fi < g.gl_pathc; fi++) {
        std::vector<char> b = loadFile(g.gl_pathv[fi]);
        if (b.size() < 8 || memcmp(b.data(), "STARM5T1", 8)) { fprintf(stderr, "bad header %s\n", g.gl_pathv[fi]); exit(2); }
        u64 o = 8;
        while (o < b.size()) {
            M5Rec r; memcpy(&r, b.data() + o, sizeof r); o += sizeof r;
            T.rb.insert(T.rb.end(), b.data() + o, b.data() + o + r.Lread); o += r.Lread; T.roff.push_back(T.rb.size());
            T.blob.insert(T.blob.end(), (uint8_t*)b.data() + o, (uint8_t*)b.data() + o + r.blobLen); o += r.blobLen; T.boff.push_back(T.blob.size());
            T.rlen.push_back(r.Lread); T.count.push_back(r.count); T.unmap.push_back((int8_t)r.unmapType); T.flags.push_back(r.flags);
            T.marker.push_back(r.mapMarker);
        }
    }
    globfree(&g);
}

static std::string blobStr(const uint8_t* p, int n) {
    std::string s; const uint8_t* e = p + n;
    while (p < e) {
        u64 u = 0; int sh = 0;
        while (p < e && (*p & 0x80)) { u |= (u64)(*p++ & 0x7f) << sh; sh += 7; }
        if (p < e) u |= (u64)(*p++) << sh;
        long long v = (long long)((u >> 1) ^ (~(u & 1) + 1));
        s += std::to_string(v); s += ' ';
    }
    return s;
}

// value of STAR's genomic length score for gLength x, with the C library's log2 (stitchWindowAligns.cpp)
static int glScore(double scale, u64 x) { return int(std::ceil(std::log2((double)x) * scale - 0.5)); }

int main(int argc, char** argv) {
    std::string gdir, truth, json, input, emit; int reps = 3, bps = 4, nthr = 128;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--genome-dir") gdir = argv[++i];
        else if (a == "--truth") truth = argv[++i];
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--json") json = argv[++i];
        else if (a == "--blocks-per-sm") bps = atoi(argv[++i]);
        else if (a == "--threads") nthr = atoi(argv[++i]);
        else if (a == "--input") input = argv[++i];     // "count sequence" lines (uniq -c): map these instead of the truth reads
        else if (a == "--emit") emit = argv[++i];       // write (sequence, result) for the sequence cache preload (inject/)
    }
    if (gdir.empty() || truth.empty()) { fprintf(stderr, "usage: m5 --genome-dir D --truth PREFIX [--json F]\n"); return 2; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device %s, %d SMs; sizeof Work<Main> %zu, Work<Heavy> %zu\n", prop.name, prop.multiProcessorCount,
            sizeof(Work<Main>), sizeof(Work<Heavy>));

    // ---- parameters
    ParamFile pf; pf.load(truth + ".params.txt");
    Params P{};
    P.maxNsplit = pf.u("maxNsplit"); P.seedSplitMin = pf.u("seedSplitMin"); P.seedSearchStartLmax = pf.u("seedSearchStartLmax");
    P.seedSearchStartLmaxOverLread = pf.d("seedSearchStartLmaxOverLread"); P.seedMapMin = pf.u("seedMapMin"); P.seedSearchLmax = pf.u("seedSearchLmax");
    P.seedMultimapNmax = pf.u("seedMultimapNmax"); P.seedPerReadNmax = pf.u("seedPerReadNmax"); P.seedPerWindowNmax = pf.u("seedPerWindowNmax");
    P.winBinChrNbits = pf.u("winBinChrNbits"); P.winBinNbits = pf.u("winBinNbits"); P.winAnchorDistNbins = pf.u("winAnchorDistNbins");
    P.winFlankNbins = pf.u("winFlankNbins"); P.winBinN = pf.u("winBinN"); P.winAnchorMultimapNmax = pf.u("winAnchorMultimapNmax");
    P.alignWindowsPerReadNmax = pf.u("alignWindowsPerReadNmax"); P.alignTranscriptsPerWindowNmax = pf.u("alignTranscriptsPerWindowNmax");
    P.alignTranscriptsPerReadNmax = pf.u("alignTranscriptsPerReadNmax");
    P.alignSJoverhangMin = pf.u("alignSJoverhangMin"); P.alignSJDBoverhangMin = pf.u("alignSJDBoverhangMin");
    P.alignSplicedMateMapLmin = pf.u("alignSplicedMateMapLmin"); P.alignSplicedMateMapLminOverLmate = pf.d("alignSplicedMateMapLminOverLmate");
    P.alignIntronMin = pf.u("alignIntronMin"); P.alignIntronMax = pf.u("alignIntronMax"); P.alignMatesGapMax = pf.u("alignMatesGapMax");
    { int k = 0; const char* s = pf.str("alignSJstitchMismatchNmax").c_str(); char* e;
      while (k < 4) { long v = strtol(s, &e, 10); if (e == s) break; P.alignSJstitchMismatchNmax[k++] = (int)v; s = e; }
      if (k != 4) { fprintf(stderr, "alignSJstitchMismatchNmax needs 4 values\n"); return 2; } }
    P.alignSoftClipAtReferenceEnds = pf.u("alignSoftClipAtReferenceEnds.yes");
    P.alignEndsTypeExt[0][0] = pf.u("alignEndsType.ext00"); P.alignEndsTypeExt[0][1] = pf.u("alignEndsType.ext01");
    P.alignEndsTypeExt[1][0] = pf.u("alignEndsType.ext10"); P.alignEndsTypeExt[1][1] = pf.u("alignEndsType.ext11");
    P.alignEndsProtrudeNbasesMax = (int)pf.i("alignEndsProtrude.nBasesMax"); P.flushRight = pf.u("alignInsertionFlush.flushRight");
    std::string im = pf.str("outFilterIntronMotifs");
    P.intronMotifsFilter = im == "None" ? 0 : im == "RemoveNoncanonical" ? 1 : im == "RemoveNoncanonicalUnannotated" ? 2 : -1;
    P.removeInconsistentStrands = pf.str("outFilterIntronStrands") == "RemoveInconsistentStrands";
    P.outSAMstrandFieldType = pf.u("outSAMstrandField.type");
    P.scoreGenomicLengthLog2scale = pf.d("scoreGenomicLengthLog2scale");
    P.outFilterMultimapScoreRange = (int)pf.i("outFilterMultimapScoreRange"); P.outFilterMultimapNmax = pf.u("outFilterMultimapNmax");
    P.multOrderRandom = pf.u("outMultimapperOrder.random"); P.outSAMmultNmax = pf.u("outSAMmultNmax");
    P.primaryAllBest = pf.str("outSAMprimaryFlag") == "AllBestScore";
    P.outFilterScoreMin = (int)pf.i("outFilterScoreMin"); P.outFilterScoreMinOverLread = pf.d("outFilterScoreMinOverLread");
    P.outFilterMatchNmin = pf.u("outFilterMatchNmin"); P.outFilterMatchNminOverLread = pf.d("outFilterMatchNminOverLread");
    P.outFilterMismatchNmax = pf.u("outFilterMismatchNmax"); P.outFilterMismatchNoverLmax = pf.d("outFilterMismatchNoverLmax");
    P.outFilterMismatchNoverReadLmax = pf.d("outFilterMismatchNoverReadLmax");
    P.scoreGap = (int)pf.i("scoreGap"); P.scoreGapNoncan = (int)pf.i("scoreGapNoncan"); P.scoreGapGCAG = (int)pf.i("scoreGapGCAG");
    P.scoreGapATAC = (int)pf.i("scoreGapATAC"); P.scoreDelBase = (int)pf.i("scoreDelBase"); P.scoreDelOpen = (int)pf.i("scoreDelOpen");
    P.scoreInsBase = (int)pf.i("scoreInsBase"); P.scoreInsOpen = (int)pf.i("scoreInsOpen"); P.scoreStitchSJshift = (int)pf.i("scoreStitchSJshift");
    P.sjdbScore = (int)pf.i("pGe.sjdbScore"); P.gSAsparseD = pf.u("pGe.gSAsparseD"); P.gSAindexNbases = pf.u("pGe.gSAindexNbases");
    // what the port does not cover (it would need the code paths named here)
    std::vector<std::string> unsupported;
    if (pf.u("readNmates") != 1) unsupported.push_back("readNmates != 1 (mates)");
    if (pf.u("pCh.segmentMin") != 0) unsupported.push_back("chimeric detection");
    if (pf.u("peOverlap.NbasesMin") != 0) unsupported.push_back("peOverlap");
    if (pf.u("Var.yes") != 0) unsupported.push_back("variation");
    if (pf.i("pGe.transform.type") > 0) unsupported.push_back("genome transform");
    if (pf.u("pGe.gType") == 101) unsupported.push_back("splice graph");
    if (pf.i("outFilterBySJoutStage") == 2) unsupported.push_back("BySJout stage 2");
    if (P.multOrderRandom || P.outSAMmultNmax != (u64)-1 || P.primaryAllBest) unsupported.push_back("multimapper order / primary flag");
    if (P.gSAsparseD > 4) unsupported.push_back("gSAsparseD > 4");
    if (P.intronMotifsFilter < 0) unsupported.push_back("outFilterIntronMotifs");
    if (P.outFilterMultimapScoreRange < 0) unsupported.push_back("negative score range");
    if (P.alignWindowsPerReadNmax > (u64)Heavy::NW || P.seedPerWindowNmax > (u64)Heavy::NWA || P.alignTranscriptsPerWindowNmax > (u64)Heavy::NWTR
        || P.seedPerReadNmax > (u64)Heavy::NP || P.maxNsplit > (u64)Heavy::NSPLIT) unsupported.push_back("STAR limits above the heavy lane's");
    if (!unsupported.empty()) { for (auto& s : unsupported) fprintf(stderr, "unsupported: %s\n", s.c_str()); return 3; }
    // genomic length score breakpoints (the value is nonincreasing in x for scale < 0)
    if (P.scoreGenomicLengthLog2scale > 0) { fprintf(stderr, "positive scoreGenomicLengthLog2scale\n"); return 3; }
    {
        double s = P.scoreGenomicLengthLog2scale;
        P.glVal0 = glScore(s, 1); P.glN = 0;
        u64 x = 1; int v = P.glVal0; const u64 XMAX = 1ULL << 40;
        while (x < XMAX && s != 0) {
            u64 lo = x, hi = x + 1;
            while (hi < XMAX && glScore(s, hi) == v) { lo = hi; hi = std::min(XMAX, hi * 2); }
            if (glScore(s, hi) == v) break;
            while (hi - lo > 1) { u64 m = lo + (hi - lo) / 2; if (glScore(s, m) == v) lo = m; else hi = m; }
            if (P.glN >= 64) { fprintf(stderr, "too many genomic length breakpoints\n"); return 3; }
            v = glScore(s, hi); P.glBreak[P.glN] = hi; P.glVal[P.glN] = v; P.glN++; x = hi;
        }
        // check against the C library on a sweep
        for (u64 xx = 1; xx < (1ULL << 24); xx = xx + 1 + xx / 997)
            if (genomicLengthScore(P, xx) != glScore(s, xx)) { fprintf(stderr, "genomic length score breakpoints wrong at %llu\n", xx); return 3; }
        fprintf(stderr, "genomic length score: %d breakpoints\n", P.glN);
    }

    // ---- genome arrays and index
    double t0 = now();
    std::vector<char> gb = loadFile(truth + ".genome.bin");
    if (memcmp(gb.data(), "STARM5G1", 8)) { fprintf(stderr, "bad genome.bin\n"); return 2; }
    u64 o = 8;
    std::vector<u64> chrStart = takeArr<u64>(gb, o), chrLength = takeArr<u64>(gb, o), chrBin = takeArr<u64>(gb, o);
    std::vector<u64> sjdbStart = takeArr<u64>(gb, o), sjdbEnd = takeArr<u64>(gb, o);
    std::vector<uint8_t> sjdbMotif = takeArr<uint8_t>(gb, o), sjdbShiftLeft = takeArr<uint8_t>(gb, o), sjdbShiftRight = takeArr<uint8_t>(gb, o);
    std::vector<uint8_t> sjdbStrand = takeArr<uint8_t>(gb, o);
    std::vector<u64> sjDstart = takeArr<u64>(gb, o), sjAstart = takeArr<u64>(gb, o);

    GenomeView H{};
    H.nGenome = fileSize(gdir + "/Genome");
    u64 nSAbyte = fileSize(gdir + "/SA");
    H.GstrandBit = pf.u("GstrandBit");
    H.GstrandMask = ~(1ULL << H.GstrandBit);
    H.saBits = H.GstrandBit + 1; H.saiBits = H.GstrandBit + 3;
    H.nSA = nSAbyte * 8 / H.saBits;
    if (H.nSA != pf.u("nSA")) { fprintf(stderr, "nSA %llu != STAR's %llu\n", H.nSA, pf.u("nSA")); }
    H.nSA = pf.u("nSA");
    if (H.nGenome != pf.u("nGenome")) { fprintf(stderr, "Genome file %llu bytes, STAR nGenome %llu\n", H.nGenome, pf.u("nGenome")); }
    H.SAiMarkNmaskC = 1ULL << (H.GstrandBit + 1); H.SAiMarkNmask = ~H.SAiMarkNmaskC; H.SAiMarkAbsentMaskC = 1ULL << (H.GstrandBit + 2);
    std::string saiPath = gdir + "/SAindex";
    u64 nb; readInto(saiPath, 0, (char*)&nb, 8);
    if (nb != P.gSAindexNbases || nb > 30) { fprintf(stderr, "gSAindexNbases mismatch\n"); return 2; }
    readInto(saiPath, 8, (char*)H.saiStart, 8 * (nb + 1));
    u64 nSAi = H.saiStart[nb], saiHdr = 8 + 8 * (nb + 1), saiBytes = (nSAi - 1) * H.saiBits / 8 + 8;
    const u64 PAD = 200;
    u64 fileG = fileSize(gdir + "/Genome");
    std::vector<char> hG(fileG + 2 * PAD + 64, 5);
    readInto(gdir + "/Genome", 0, hG.data() + PAD, fileG);
    std::vector<u64> hSA(nSAbyte / 8 + 4, 0); readInto(gdir + "/SA", 0, (char*)hSA.data(), nSAbyte);
    std::vector<u64> hSAi(saiBytes / 8 + 4, 0); readInto(saiPath, saiHdr, (char*)hSAi.data(), saiBytes);
    H.nGenome = pf.u("nGenome");
    H.G = hG.data() + PAD; H.SA = hSA.data(); H.SAi = hSAi.data();
    H.chrStart = chrStart.data(); H.chrLength = chrLength.data(); H.chrBin = chrBin.data(); H.nChr = chrLength.size(); H.chrBinN = chrBin.size();
    H.sjGstart = pf.u("sjGstart"); H.sjdbOverhang = pf.u("sjdbOverhang"); H.sjdbLength = pf.u("sjdbLength"); H.sjdbN = pf.u("sjdbN");
    H.sjdbStart = sjdbStart.data(); H.sjdbEnd = sjdbEnd.data(); H.sjDstart = sjDstart.data(); H.sjAstart = sjAstart.data();
    H.sjdbMotif = sjdbMotif.data(); H.sjdbShiftLeft = sjdbShiftLeft.data(); H.sjdbShiftRight = sjdbShiftRight.data(); H.sjdbStrand = sjdbStrand.data();
    GenomeView D = H;
    char* dG = toDev(hG); D.G = dG + PAD;
    D.SA = toDev(hSA); D.SAi = toDev(hSAi);
    D.chrStart = toDev(chrStart); D.chrLength = toDev(chrLength); D.chrBin = toDev(chrBin);
    D.sjdbStart = toDev(sjdbStart); D.sjdbEnd = toDev(sjdbEnd); D.sjDstart = toDev(sjDstart); D.sjAstart = toDev(sjAstart);
    D.sjdbMotif = toDev(sjdbMotif); D.sjdbShiftLeft = toDev(sjdbShiftLeft); D.sjdbShiftRight = toDev(sjdbShiftRight); D.sjdbStrand = toDev(sjdbStrand);
    double tIndex = now() - t0;
    fprintf(stderr, "index + genome arrays loaded in %.1f s; sjdbN %llu, chr %llu\n", tIndex, H.sjdbN, H.nChr);

    // ---- truth
    t0 = now();
    Truth T;
    if (input.empty()) loadTruth(truth, T);
    else {   // no STAR truth: the sequences of a whole sample
        FILE* f = fopen(input.c_str(), "r"); if (!f) { perror(input.c_str()); return 2; }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            char* q = line; while (*q == ' ') q++;
            uint32_t c = (uint32_t)strtoul(q, &q, 10); while (*q == ' ') q++;
            int L = 0;
            for (; q[L] && q[L] != '\n' && q[L] != '\r'; L++) {
                char b = q[L]; T.rb.push_back(b == 'A' ? 0 : b == 'C' ? 1 : b == 'G' ? 2 : b == 'T' ? 3 : 4);
            }
            T.roff.push_back(T.rb.size()); T.rlen.push_back((uint16_t)L); T.count.push_back(c);
            T.boff.push_back(0); T.unmap.push_back(0); T.flags.push_back(0); T.marker.push_back(0);
        }
        fclose(f);
    }
    u64 n = T.rlen.size(), nReadsW = 0; for (auto c : T.count) nReadsW += c;
    fprintf(stderr, "truth: %llu distinct reads (%llu reads), loaded in %.1f s\n", n, nReadsW, now() - t0);

    // ---- GPU main lane
    char* dRb = toDev(T.rb); u64* dRoff = toDev(T.roff); uint16_t* dRlen = toDev(T.rlen);
    uint8_t* dOut; CK(cudaMalloc(&dOut, n * (u64)Main::NBLOB));
    uint16_t* dOutLen; CK(cudaMalloc(&dOutLen, n * 2)); uint8_t* dStatus; CK(cudaMalloc(&dStatus, n)); uint8_t* dWhy; CK(cudaMalloc(&dWhy, n));
    int8_t* dUnmap; CK(cudaMalloc(&dUnmap, n));
    int blocks = prop.multiProcessorCount * bps; u64 nt = (u64)blocks * nthr;
    Work<Main>* dWork; CK(cudaMalloc(&dWork, nt * sizeof(Work<Main>)));
    ReadBuf<Main::LREAD>* dRbuf; CK(cudaMalloc(&dRbuf, nt * sizeof(ReadBuf<Main::LREAD>)));
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));
    fprintf(stderr, "main lane: %d blocks x %d threads, work %.2f GB\n", blocks, nthr, nt * (double)sizeof(Work<Main>) / 1e9);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<double> ms;
    for (int r = 0; r < reps + 1; r++) {
        cudaEventRecord(e0);
        k_map<<<blocks, nthr>>>(P, D, dRb, dRoff, dRlen, n, dWork, dRbuf, dOut, dOutLen, dStatus, dWhy, dUnmap);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float m; cudaEventElapsedTime(&m, e0, e1); if (r > 0) ms.push_back(m);
        fprintf(stderr, "  run %d: %.1f ms\n", r, m);
    }
    double gpuMs = median(ms);
    std::vector<uint8_t> out(n * (u64)Main::NBLOB), status(n), why(n); std::vector<uint16_t> outLen(n); std::vector<int8_t> unmap(n);
    CK(cudaMemcpy(out.data(), dOut, out.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(outLen.data(), dOutLen, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(status.data(), dStatus, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(why.data(), dWhy, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(unmap.data(), dUnmap, n, cudaMemcpyDeviceToHost));

    // ---- heavy lane (host, STAR's limits)
    std::vector<u64> heavy; u64 whyCount[16] = {0};
    for (u64 i = 0; i < n; i++) if (status[i] != (uint8_t)Status::Ok) { heavy.push_back(i); whyCount[why[i] & 15]++; }
    std::vector<std::vector<uint8_t>> hout(heavy.size()); std::vector<int8_t> hunmap(heavy.size()); std::vector<uint8_t> hstatus(heavy.size());
    t0 = now();
    int nHostThreads = omp_get_max_threads();
    #pragma omp parallel
    {
        Work<Heavy>* w = new Work<Heavy>;
        ReadBuf<Heavy::LREAD>* b = new ReadBuf<Heavy::LREAD>;
        std::vector<uint8_t> buf(Heavy::NBLOB);
        #pragma omp for schedule(dynamic, 16)
        for (long long k = 0; k < (long long)heavy.size(); k++) {
            u64 i = heavy[k];
            Blob bl{buf.data(), 0, Heavy::NBLOB, false};
            Status s = mapRead(P, H, *w, *b, T.rb.data() + T.roff[i], T.rlen[i], bl);
            hstatus[k] = (uint8_t)s; hunmap[k] = (int8_t)w->unmapType; hout[k].assign(buf.data(), buf.data() + bl.n);
        }
        delete w; delete b;
    }
    double heavySec = now() - t0;

    // ---- compare
    u64 same = 0, diff = 0, sameW = 0, diffW = 0, heavyFail = 0, nPrinted = 0, sameUnmapOnly = 0;
    std::vector<u64> heavyIdx(n, ~0ULL); for (u64 k = 0; k < heavy.size(); k++) heavyIdx[heavy[k]] = k;
    u64 classN[5] = {0}, classSame[5] = {0};   // by STAR result: unmapped(0..3 -> idx) / mapped unique / mapped multi
    for (u64 i = 0; i < n; i++) {
        const uint8_t* ours; int olen; int8_t oun;
        if (heavyIdx[i] == ~0ULL) { ours = out.data() + i * Main::NBLOB; olen = outLen[i]; oun = unmap[i]; }
        else {
            u64 k = heavyIdx[i];
            if (hstatus[k] != (uint8_t)Status::Ok) heavyFail++;
            ours = hout[k].data(); olen = (int)hout[k].size(); oun = hunmap[k];
        }
        const uint8_t* tb = T.blob.data() + T.boff[i]; int tlen = (int)(T.boff[i + 1] - T.boff[i]);
        bool eq = olen == tlen && memcmp(ours, tb, tlen) == 0 && oun == T.unmap[i];
        int cls = T.unmap[i] >= 0 ? 0 : 1;
        classN[cls]++; if (eq) classSame[cls]++;
        if (eq) { same++; sameW += T.count[i]; }
        else {
            diff++; diffW += T.count[i];
            if (olen == tlen && memcmp(ours, tb, tlen) == 0) sameUnmapOnly++;
            if (nPrinted < 40) {
                nPrinted++;
                std::string seq; for (int k = 0; k < T.rlen[i]; k++) seq += "ACGTN"[std::min(4, (int)T.rb[T.roff[i] + k])];
                fprintf(stderr, "DIFF read %llu L=%u count=%u flags=%u marker=%u lane=%s unmap star=%d ours=%d\n  seq  %s\n  star %s\n  ours %s\n",
                        i, T.rlen[i], T.count[i], T.flags[i], T.marker[i], heavyIdx[i] == ~0ULL ? "main" : "heavy", T.unmap[i], oun,
                        seq.c_str(), blobStr(tb, tlen).c_str(), blobStr(ours, olen).c_str());
            }
        }
    }
    if (!emit.empty()) {   // header, then per sequence without N: u16 L, L bases, u32 blobLen, blob
        FILE* f = fopen(emit.c_str(), "wb"); fwrite("M5BLOBS1", 8, 1, f); u64 nEmit = 0;
        for (u64 i = 0; i < n; i++) {
            bool hasN = false; for (int k = 0; k < T.rlen[i]; k++) if (T.rb[T.roff[i] + k] > 3) hasN = true;
            if (hasN) continue;
            const uint8_t* ours; uint32_t olen;
            if (heavyIdx[i] == ~0ULL) { ours = out.data() + i * Main::NBLOB; olen = outLen[i]; }
            else { if (hstatus[heavyIdx[i]] != (uint8_t)Status::Ok) continue; ours = hout[heavyIdx[i]].data(); olen = (uint32_t)hout[heavyIdx[i]].size(); }
            uint16_t L = T.rlen[i]; fwrite(&L, 2, 1, f); fwrite(T.rb.data() + T.roff[i], 1, L, f); fwrite(&olen, 4, 1, f); fwrite(ours, 1, olen, f); nEmit++;
        }
        fclose(f); fprintf(stderr, "emitted %llu sequence results to %s\n", nEmit, emit.c_str());
    }
    double rate = n / (gpuMs / 1e3);
    fprintf(stderr, "\nM5: %llu / %llu distinct reads identical (%.6f%%), reads weighted %llu / %llu (%.6f%%); differ %llu (unmapType only %llu)\n",
            same, n, 100.0 * same / n, sameW, nReadsW, 100.0 * sameW / nReadsW, diff, sameUnmapOnly);
    fprintf(stderr, "    unmapped by STAR: %llu (%llu identical), mapped: %llu (%llu identical)\n", classN[0], classSame[0], classN[1], classSame[1]);
    fprintf(stderr, "    GPU main lane %.1f ms for %llu reads = %.1f M distinct reads/s (incl. heavy-lane reads that stop early)\n", gpuMs, n, rate / 1e6);
    fprintf(stderr, "    heavy lane: %zu reads (%.4f%%), %.2f s on %d host threads, failures %llu; why:", heavy.size(), 100.0 * heavy.size() / n,
            heavySec, nHostThreads, heavyFail);
    const char* whyName[] = {"none", "lread", "split", "pc", "win", "wa", "winbin", "exons", "wtr", "keep", "blob"};
    for (int k = 0; k < 11; k++) if (whyCount[k]) fprintf(stderr, " %s=%llu", whyName[k], whyCount[k]);
    fprintf(stderr, "\n");
    if (!json.empty()) {
        FILE* f = fopen(json.c_str(), "w");
        fprintf(f, "{\n  \"device\": \"%s\",\n  \"distinct_reads\": %llu,\n  \"reads\": %llu,\n  \"identical_distinct\": %llu,\n  \"identical_reads\": %llu,\n"
                   "  \"differ_distinct\": %llu,\n  \"differ_reads\": %llu,\n  \"star_unmapped\": %llu,\n  \"star_unmapped_identical\": %llu,\n"
                   "  \"star_mapped\": %llu,\n  \"star_mapped_identical\": %llu,\n  \"gpu_ms_median\": %.3f,\n  \"gpu_ms_runs\": [",
                prop.name, n, nReadsW, same, sameW, diff, diffW, classN[0], classSame[0], classN[1], classSame[1], gpuMs);
        for (size_t k = 0; k < ms.size(); k++) fprintf(f, "%s%.3f", k ? ", " : "", ms[k]);
        fprintf(f, "],\n  \"gpu_distinct_reads_per_s\": %.0f,\n  \"heavy_reads\": %zu,\n  \"heavy_host_s\": %.3f,\n  \"heavy_host_threads\": %d,\n  \"heavy_failures\": %llu,\n  \"heavy_why\": {",
                rate, heavy.size(), heavySec, nHostThreads, heavyFail);
        bool first = true; for (int k = 0; k < 11; k++) if (whyCount[k]) { fprintf(f, "%s\"%s\": %llu", first ? "" : ", ", whyName[k], whyCount[k]); first = false; }
        fprintf(f, "},\n  \"blocks\": %d,\n  \"threads_per_block\": %d,\n  \"work_bytes_per_thread\": %zu,\n  \"index_load_s\": %.1f\n}\n", blocks, nthr, sizeof(Work<Main>), tIndex);
        fclose(f);
    }
    return 0;
}
