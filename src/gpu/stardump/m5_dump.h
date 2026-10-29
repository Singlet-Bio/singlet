// W11 M5 truth: per mapped read, the sequence STAR mapped and its full mapping result; once per process, the alignment
// parameters and genome arrays that seed storage, windows, stitching and multimapper selection read.
// Used only by ReadAlign_oneRead_dump.cpp in the `stardump` executable (STARDUMP_M5=<prefix>).
//
// Files:
//   <prefix>.<tid>.bin   header "STARM5T1", then records: M5Rec, Read1[0][Lread] (numeric bases), blob[blobLen]
//   <prefix>.params.txt  "name value..." lines (every scalar the port reads)
//   <prefix>.genome.bin  "STARM5G1", then arrays (u64 count, then the elements) in the order written by params()
//
// The blob is the sequence cache's encoding (ReadAlign_seqCache.cpp: nW, nTr, iBest, [trInit rLength], then per
// transcript nExons, the scalars, the exon rows), with its refusals turned into flags instead.
#pragma once
#include "ReadAlign.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>

namespace stardump_m5 {

#pragma pack(push, 1)
struct M5Rec {
    uint64_t iReadAll;
    uint32_t count;      // multiplicity of the distinct sequence (read name "..._<n>")
    uint16_t Lread;
    uint8_t flags;       // bit0: cache-encodable (seqCacheEncode would accept), bit1: sequence has N, bit2: chimRecord, bit3: peOv
    uint8_t pad;
    int32_t unmapType;
    uint32_t mapMarker;
    uint32_t blobLen;
};
#pragma pack(pop)

inline const char* prefix() { static const char* p = getenv("STARDUMP_M5"); return p; }
inline bool enabled() { return prefix() != nullptr; }

struct Writer {
    FILE* f = nullptr;
    std::vector<unsigned char> buf;
    void put(const void* p, size_t n) {
        const unsigned char* c = (const unsigned char*)p;
        buf.insert(buf.end(), c, c + n);
        if (buf.size() > (8u << 20)) flush();
    }
    void flush() { if (f && !buf.empty()) fwrite(buf.data(), 1, buf.size(), f); buf.clear(); }
};
inline std::mutex& mu() { static std::mutex m; return m; }
inline std::vector<Writer*>& writers() { static std::vector<Writer*> v; return v; }
inline void closeAll() {
    std::lock_guard<std::mutex> lk(mu());
    for (Writer* w : writers()) { w->flush(); if (w->f) fclose(w->f); w->f = nullptr; }
}
inline Writer* writer() {
    static thread_local Writer* w = nullptr;
    if (w == nullptr) {
        w = new Writer;
        char path[4096];
        snprintf(path, sizeof path, "%s.%ld.bin", prefix(), (long)syscall(SYS_gettid));
        w->f = fopen(path, "wb");
        if (w->f == nullptr) { perror(path); exit(3); }
        w->put("STARM5T1", 8);
        std::lock_guard<std::mutex> lk(mu());
        writers().push_back(w);
    }
    return w;
}

inline void putV(std::vector<uint8> &b, int64 v) {//as ReadAlign_seqCache.cpp
    uint64 u = ((uint64) v << 1) ^ (uint64) (v >> 63);
    while (u >= 0x80) {b.push_back((uint8) (u | 0x80)); u >>= 7;}
    b.push_back((uint8) u);
}
//field order of ReadAlign_seqCache.cpp forEachScalar / forEachRow
inline void putScalars(std::vector<uint8> &b, Transcript &t) {
    int64 v[] = {(int64)t.rStart, (int64)t.roStart, (int64)t.rLength, (int64)t.gStart, (int64)t.gLength, (int64)t.cStart,
                 (int64)t.Chr, (int64)t.Str, (int64)t.roStr,
                 (int64)t.nMatch, (int64)t.nMM, (int64)t.mappedLength, (int64)t.extendL,
                 (int64)t.nGap, (int64)t.lGap, (int64)t.nDel, (int64)t.nIns, (int64)t.lDel, (int64)t.lIns, (int64)t.nUnique, (int64)t.nAnchor,
                 (int64)t.intronMotifs[0], (int64)t.intronMotifs[1], (int64)t.intronMotifs[2],
                 (int64)t.maxScore, (int64)t.iFrag, (int64)t.sjMotifStrand, (int64)(t.sjYes ? 1 : 0), (int64)(t.primaryFlag ? 1 : 0)};
    for (int64 x : v) putV(b, x);
}
inline void putRow(std::vector<uint8> &b, Transcript &t, uint64 iex) {
    for (int k=0; k<EX_SIZE; k++) putV(b, (int64)t.exons[iex][k]);
    if (iex+1 < t.nExons) {
        putV(b, t.canonSJ[iex]); putV(b, t.sjAnnot[iex]); putV(b, t.sjStr[iex]);
        if (t.canonSJ[iex] >= 0) {putV(b, (int64)t.shiftSJ[iex][0]); putV(b, (int64)t.shiftSJ[iex][1]);};
    };
}

inline void record(bool encodable, bool chimRecord, bool peOvYes, uint64 iReadAll, const char* readName, uint64 Lread, const char* R, uint nW, uint64 nTr,
                   Transcript** trMult, Transcript* trBest, Transcript* trInit, int unmapType, uint mapMarker) {
    std::vector<uint8> b;
    uint8_t flags = 0;
    bool hasN = false;
    for (uint64 i=0; i<Lread; i++) if ((uint8)R[i] > 3) hasN = true;
    if (encodable) flags |= 1;
    if (hasN) flags |= 2;
    if (chimRecord) flags |= 4;
    if (peOvYes) flags |= 8;
    int64 iBest = -1;
    if (nW > 0) for (uint64 i=0; i<nTr; i++) if (trMult[i] == trBest) {iBest = (int64)i; break;}
    putV(b, (int64)nW); putV(b, (int64)nTr); putV(b, iBest);
    if (nW == 0) putV(b, (int64)trInit->rLength);
    for (uint64 i=0; i<nTr; i++) {
        Transcript &t = *trMult[i];
        putV(b, (int64)t.nExons);
        putScalars(b, t);
        for (uint64 iex=0; iex<t.nExons && iex<MAX_N_EXONS; iex++) putRow(b, t, iex);
    };
    M5Rec r;
    r.iReadAll = iReadAll;
    const char* u = readName ? strrchr(readName, '_') : nullptr;
    r.count = u ? (uint32_t)strtoul(u + 1, nullptr, 10) : 1;
    r.Lread = (uint16_t)Lread; r.flags = flags; r.pad = 0;
    r.unmapType = unmapType; r.mapMarker = mapMarker; r.blobLen = (uint32_t)b.size();
    Writer* w = writer();
    w->put(&r, sizeof r); w->put(R, Lread); w->put(b.data(), b.size());
}

template <class T> inline void arr(FILE* f, const T* a, uint64 n) { fwrite(&n, 8, 1, f); if (n) fwrite(a, sizeof(T), n, f); }

inline void params(Parameters &P, Genome &G) {
    static std::once_flag once;
    std::call_once(once, [&]() {
        std::string pfx = prefix();
        FILE* f = fopen((pfx + ".params.txt").c_str(), "w");
        if (!f) { perror("params"); exit(3); }
        auto u = [&](const char* n, uint64 v) { fprintf(f, "%s %llu\n", n, (unsigned long long)v); };
        auto i = [&](const char* n, int64 v) { fprintf(f, "%s %lld\n", n, (long long)v); };
        auto d = [&](const char* n, double v) { fprintf(f, "%s %.17g\n", n, v); };
        auto s = [&](const char* n, const std::string& v) { fprintf(f, "%s %s\n", n, v.c_str()); };
        u("readNmates", P.readNmates); u("readNends", P.readNends);
        u("maxNsplit", P.maxNsplit); u("seedSplitMin", P.seedSplitMin); u("seedSearchStartLmax", P.seedSearchStartLmax);
        d("seedSearchStartLmaxOverLread", P.seedSearchStartLmaxOverLread); u("seedMapMin", P.seedMapMin); u("seedSearchLmax", P.seedSearchLmax);
        u("seedMultimapNmax", P.seedMultimapNmax); u("seedPerReadNmax", P.seedPerReadNmax); u("seedPerWindowNmax", P.seedPerWindowNmax);
        u("seedNoneLociPerWindow", P.seedNoneLociPerWindow);
        u("winBinChrNbits", P.winBinChrNbits); u("winBinNbits", P.winBinNbits); u("winAnchorDistNbins", P.winAnchorDistNbins);
        u("winFlankNbins", P.winFlankNbins); u("winBinN", P.winBinN); u("winAnchorMultimapNmax", P.winAnchorMultimapNmax);
        u("alignWindowsPerReadNmax", P.alignWindowsPerReadNmax); u("alignTranscriptsPerWindowNmax", P.alignTranscriptsPerWindowNmax);
        u("alignTranscriptsPerReadNmax", P.alignTranscriptsPerReadNmax);
        u("alignSJoverhangMin", P.alignSJoverhangMin); u("alignSJDBoverhangMin", P.alignSJDBoverhangMin);
        u("alignSplicedMateMapLmin", P.alignSplicedMateMapLmin); d("alignSplicedMateMapLminOverLmate", P.alignSplicedMateMapLminOverLmate);
        u("alignIntronMin", P.alignIntronMin); u("alignIntronMax", P.alignIntronMax); u("alignMatesGapMax", P.alignMatesGapMax);
        fprintf(f, "alignSJstitchMismatchNmax"); for (auto x : P.alignSJstitchMismatchNmax) fprintf(f, " %d", (int)x); fprintf(f, "\n");
        u("alignSoftClipAtReferenceEnds.yes", P.alignSoftClipAtReferenceEnds.yes);
        u("alignEndsType.ext00", P.alignEndsType.ext[0][0]); u("alignEndsType.ext01", P.alignEndsType.ext[0][1]);
        u("alignEndsType.ext10", P.alignEndsType.ext[1][0]); u("alignEndsType.ext11", P.alignEndsType.ext[1][1]);
        i("alignEndsProtrude.nBasesMax", P.alignEndsProtrude.nBasesMax); u("alignInsertionFlush.flushRight", P.alignInsertionFlush.flushRight);
        s("outFilterIntronMotifs", P.outFilterIntronMotifs); s("outFilterIntronStrands", P.outFilterIntronStrands);
        u("outSAMstrandField.type", P.outSAMstrandField.type); i("outFilterBySJoutStage", P.outFilterBySJoutStage);
        d("scoreGenomicLengthLog2scale", P.scoreGenomicLengthLog2scale);
        i("outFilterMultimapScoreRange", P.outFilterMultimapScoreRange); u("outFilterMultimapNmax", P.outFilterMultimapNmax);
        u("outMultimapperOrder.random", P.outMultimapperOrder.random); u("outSAMmultNmax", P.outSAMmultNmax); s("outSAMprimaryFlag", P.outSAMprimaryFlag);
        i("outFilterScoreMin", P.outFilterScoreMin); d("outFilterScoreMinOverLread", P.outFilterScoreMinOverLread);
        u("outFilterMatchNmin", P.outFilterMatchNmin); d("outFilterMatchNminOverLread", P.outFilterMatchNminOverLread);
        u("outFilterMismatchNmax", P.outFilterMismatchNmax); d("outFilterMismatchNoverLmax", P.outFilterMismatchNoverLmax);
        d("outFilterMismatchNoverReadLmax", P.outFilterMismatchNoverReadLmax);
        i("scoreGap", P.scoreGap); i("scoreGapNoncan", P.scoreGapNoncan); i("scoreGapGCAG", P.scoreGapGCAG); i("scoreGapATAC", P.scoreGapATAC);
        i("scoreDelBase", P.scoreDelBase); i("scoreDelOpen", P.scoreDelOpen); i("scoreInsBase", P.scoreInsBase); i("scoreInsOpen", P.scoreInsOpen);
        i("scoreStitchSJshift", P.scoreStitchSJshift);
        u("pCh.segmentMin", P.pCh.segmentMin); u("peOverlap.yes", P.peOverlap.yes); u("peOverlap.NbasesMin", P.peOverlap.NbasesMin);
        i("pGe.sjdbScore", P.pGe.sjdbScore); u("pGe.gSAsparseD", P.pGe.gSAsparseD); u("pGe.gSAindexNbases", P.pGe.gSAindexNbases);
        u("pGe.gType", P.pGe.gType); i("pGe.transform.type", P.pGe.transform.type);
        u("Var.yes", G.Var != nullptr && G.Var->yes);
        u("nGenome", G.nGenome); u("nSA", G.nSA); u("GstrandBit", G.GstrandBit); u("chrBinN", G.chrBinN);
        u("genomeChrBinNbases", G.genomeChrBinNbases); u("nChrReal", G.nChrReal);
        u("sjdbOverhang", G.sjdbOverhang); u("sjdbLength", G.sjdbLength); u("sjChrStart", G.sjChrStart); u("sjdbN", G.sjdbN); u("sjGstart", G.sjGstart);
        u("MAX_N_EXONS", MAX_N_EXONS); u("DEF_readSeqLengthMax", DEF_readSeqLengthMax);
        fclose(f);
        FILE* g = fopen((pfx + ".genome.bin").c_str(), "wb");
        if (!g) { perror("genome.bin"); exit(3); }
        fwrite("STARM5G1", 8, 1, g);
        arr(g, G.chrStart.data(), (uint64)G.chrStart.size());
        arr(g, G.chrLength.data(), (uint64)G.chrLength.size());
        arr(g, G.chrBin, (uint64)G.chrBinN);
        arr(g, G.sjdbStart, (uint64)G.sjdbN); arr(g, G.sjdbEnd, (uint64)G.sjdbN);
        arr(g, G.sjdbMotif, (uint64)G.sjdbN); arr(g, G.sjdbShiftLeft, (uint64)G.sjdbN); arr(g, G.sjdbShiftRight, (uint64)G.sjdbN);
        arr(g, G.sjdbStrand, (uint64)G.sjdbN); arr(g, G.sjDstart, (uint64)G.sjdbN); arr(g, G.sjAstart, (uint64)G.sjdbN);
        fclose(g);
    });
}

} // namespace stardump_m5
