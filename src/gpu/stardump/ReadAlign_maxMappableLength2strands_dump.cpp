// STAR seed-search dump (W11, docs/gpu-engine/exploration.md §9 "The STAR dump").
//
// A copy of src/star/ReadAlign_maxMappableLength2strands.cpp (vendored STAR 2.7.11b, prod/campaign 28969b87) that is
// linked INSTEAD of the original into the `stardump` executable. The search logic is unchanged, line for line; the
// SA_BOUNDARY_PREFETCH hint and the STAR_PROFILE counters (both off in the campaign build, no effect on results) are
// dropped. The only additions are the two dump blocks, which record every call STAR makes (inputs and the result of each iDist) to a
// per-thread binary file. Layout: dump_format.h. Enabled when STARDUMP_PREFIX is set; otherwise behaves as the original.
#include "ReadAlign.h"
#include "SuffixArrayFuns.h"
#include "ErrorWarning.h"
#include "StarProfile.h"
#include "dump_format.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>

namespace stardump {

struct Writer {
    FILE* f = nullptr;
    std::vector<unsigned char> buf;
    uint64 lastRead = ~0ULL;
    void put(const void* p, size_t n) {
        const unsigned char* c = (const unsigned char*)p;
        buf.insert(buf.end(), c, c + n);
        if (buf.size() > (8u << 20)) flush();
    }
    void flush() {
        if (f && !buf.empty()) fwrite(buf.data(), 1, buf.size(), f);
        buf.clear();
    }
};

static std::mutex g_mu;
static std::vector<Writer*> g_writers;

void closeAll() {
    std::lock_guard<std::mutex> lk(g_mu);
    for (Writer* w : g_writers) {
        w->flush();
        if (w->f) fclose(w->f);
        w->f = nullptr;
    }
}

static Writer* writer() {
    static thread_local Writer* w = nullptr;
    static const char* prefix = getenv("STARDUMP_PREFIX");
    if (prefix == nullptr) return nullptr;
    if (w == nullptr) {
        w = new Writer;
        char path[4096];
        snprintf(path, sizeof path, "%s.%ld.bin", prefix, (long)syscall(SYS_gettid));
        w->f = fopen(path, "wb");
        if (w->f == nullptr) { perror(path); exit(3); }
        FileHeader h;
        w->put(&h, sizeof h);
        std::lock_guard<std::mutex> lk(g_mu);
        g_writers.push_back(w);
    }
    return w;
}

// multiplicity of a distinct sequence, carried in the read name as "..._<count>"
static uint32_t nameCount(const char* name) {
    const char* u = name ? strrchr(name, '_') : nullptr;
    return u ? (uint32_t)strtoul(u + 1, nullptr, 10) : 1;
}

} // namespace stardump

uint ReadAlign::maxMappableLength2strands(uint pieceStartIn, uint pieceLengthIn, uint iDir, uint iSA1, uint iSA2, uint& maxLbest, uint iFrag) {
    //returns number of mappings, maxMappedLength=mapped length
    uint Nrep=0, indStartEnd[2], maxL;

    uint NrepAll[P.pGe.gSAsparseD], indStartEndAll[P.pGe.gSAsparseD][2], maxLall[P.pGe.gSAsparseD];
    uint8_t branchAll[P.pGe.gSAsparseD], LindAll[P.pGe.gSAsparseD];
    maxLbest=0;

    bool dirR = iDir==0;

    stardump::Writer* dw = stardump::writer();
    if (dw && dw->lastRead != iReadAll) {//first seed call of this read: record the read
        dw->lastRead = iReadAll;
        stardump::ReadRec r;
        r.iReadAll = iReadAll;
        r.count = stardump::nameCount(readName);
        r.Lread = (uint16_t)Lread;
        r.Nsplit = (uint8_t)Nsplit;
        dw->put(&r, sizeof r);
        for (uint ip = 0; ip < Nsplit; ip++) {
            uint16_t sp[3] = {(uint16_t)splitR[0][ip], (uint16_t)splitR[1][ip], (uint16_t)splitR[2][ip]};
            dw->put(sp, sizeof sp);
        }
        dw->put(Read1[0], Lread);
        dw->put(Read1[1], Lread);
    }

    for (uint iDist=0; iDist<min(pieceLengthIn,P.pGe.gSAsparseD); iDist++) {//cycle through different distances
        uint pieceStart;
        uint pieceLength=pieceLengthIn-iDist;

        //calculate full index
        uint Lmax=min(P.pGe.gSAindexNbases,pieceLength);
        uint ind1=0;
        if (dirR) {//forward search
            pieceStart=pieceStartIn+iDist;
            for (uint ii=0;ii<Lmax;ii++) {//calculate index TODO: make the index calculation once for the whole read and store it
                ind1 <<=2LLU;
                ind1 += ((uint) Read1[0][pieceStart+ii]);
            };
        } else {//reverse search
            pieceStart=pieceStartIn-iDist;
            for (uint ii=0;ii<Lmax;ii++) {//calculate index TODO: make the index calculation once for the whole read and store it
                ind1 <<=2LLU;
                ind1 += ( 3-((uint) Read1[0][pieceStart-ii]) );
            };
        };

        //find SA boundaries
        uint Lind=Lmax;
        while (Lind>0) {//check the presence of the prefix for Lind
            iSA1=mapGen.SAi[mapGen.genomeSAindexStart[Lind-1]+ind1]; // starting point for suffix array search.
            if ((iSA1 & mapGen.SAiMarkAbsentMaskC) == 0) {//prefix exists
                break;
            } else {//this prefix does not exist, reduce Lind
                --Lind;
                ind1 = ind1 >> 2;
            };
        };

        // define upper bound for suffix array range search.
        bool iSA2good = true;
        if (mapGen.genomeSAindexStart[Lind-1]+ind1+1 < mapGen.genomeSAindexStart[Lind]) {//we are not at the end of the SA
            iSA2 = mapGen.SAi[mapGen.genomeSAindexStart[Lind-1]+ind1+1];
            if ( (iSA2 & mapGen.SAiMarkAbsentMaskC) == 0) {
                iSA2 = (iSA2 & mapGen.SAiMarkNmask) - 1;
            } else {
                iSA2 = mapGen.nSA-1; //safe, but can probably do better
                iSA2good = false;
            };
        } else {
            iSA2=mapGen.nSA-1;
            iSA2good = false;
        };

        uint8_t branch;
        bool iSA1noN = (iSA1 & mapGen.SAiMarkNmaskC)==0;
        if (Lind < P.pGe.gSAindexNbases && iSA1noN && iSA2good) {//no need for SA search
            branch = stardump::BR_SAI_ONLY;
            indStartEnd[0]=iSA1;
            indStartEnd[1]=iSA2;
            Nrep=indStartEnd[1]-indStartEnd[0]+1;
            maxL=Lind;
        } else if (iSA1==iSA2 && iSA1noN && iSA2good) {//unique align already, just find maxL
            branch = stardump::BR_UNIQUE_EXTEND;
            if ((iSA1 & mapGen.SAiMarkNmaskC)!=0) {
                ostringstream errOut;
                errOut  << "BUG: in ReadAlign::maxMappableLength2strands";
                exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_BUG, P);
            };
            indStartEnd[0]=indStartEnd[1]=iSA1;
            Nrep=1;
            bool comparRes;
            maxL=compareSeqToGenome(mapGen, Read1, pieceStart, pieceLength, Lind, iSA1, dirR, comparRes);
        } else {//need SA search, pieceLength>maxL
            branch = stardump::BR_SA_SEARCH;
            if (iSA2good && iSA1noN) {
                maxL = Lind; //Lind bases were already matched
            } else {
                maxL=0;
            };
            Nrep = maxMappableLength(mapGen, Read1, pieceStart, pieceLength, iSA1 & mapGen.SAiMarkNmask, iSA2, dirR, maxL, indStartEnd);
        };

        if (maxL+iDist > maxLbest) {//this idist is better
            maxLbest=maxL+iDist;
        };
        NrepAll[iDist]=Nrep;
        indStartEndAll[iDist][0]=indStartEnd[0];
        indStartEndAll[iDist][1]=indStartEnd[1];
        maxLall[iDist]=maxL;
        branchAll[iDist]=branch;
        LindAll[iDist]=(uint8_t)Lind;
    };

    if (dw) {//one record per iDist (gSAsparseD = 1 for the Singlet indexes: one record per call)
        for (uint iDist=0; iDist<min(pieceLengthIn,P.pGe.gSAsparseD); iDist++) {
            stardump::SeedRec s;
            s.iDir = (uint8_t)iDir;
            s.iFrag = (uint8_t)iFrag;
            s.branch = branchAll[iDist];
            s.iDist = (uint8_t)iDist;
            s.pieceStart = (uint16_t)pieceStartIn;
            s.pieceLength = (uint16_t)pieceLengthIn;
            s.Lind = LindAll[iDist];
            s.maxL = (uint16_t)maxLall[iDist];
            s.Nrep = NrepAll[iDist];
            s.ind0 = indStartEndAll[iDist][0];
            s.ind1 = indStartEndAll[iDist][1];
            s.sa0 = NrepAll[iDist] > 0 ? (uint64_t)mapGen.SA[indStartEndAll[iDist][0]] : ~0ULL;
            dw->put(&s, sizeof s);
        }
    }

    for (uint iDist=0; iDist<min(pieceLengthIn,P.pGe.gSAsparseD); iDist++) {//cycle through different distances, store the ones with largest maxL
        if ( (maxLall[iDist]+iDist) == maxLbest) {
            storeAligns(iDir, (dirR ? pieceStartIn+iDist : pieceStartIn-iDist), NrepAll[iDist], maxLall[iDist], indStartEndAll[iDist], iFrag);
        };
    };
    return Nrep;
};
