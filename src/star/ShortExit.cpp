// Exact early exit for "unmapped: too short" reads: see ShortExit.h for the argument.

#include "ShortExit.h"
#include "ReadAlign.h"
#include "Parameters.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

ShortExit *g_shortExit = nullptr;

ShortExit *ShortExit::create(Parameters &P) {
    const char *e = getenv("SINGLET_STAR_SHORTEXIT");
    int mode = 0;
    if (e != nullptr) {
        if (strcmp(e, "1") == 0 || strcmp(e, "on") == 0) mode = 1;
        else if (strcmp(e, "verify") == 0 || strcmp(e, "2") == 0) mode = 2;
    }
    if (mode == 0)
        return nullptr;
    //the outcome of a proven read must depend on unmapType only, and stitching must be per window
    std::string why;
    if (P.readNmates != 1) why = "paired-end alignment";
    else if (P.pGe.gType == 101) why = "splice-graph genome";
    else if (P.pGe.transform.type != 0) why = "genome transformation";
    else if (P.pCh.segmentMin > 0) why = "chimeric detection";
    else if (P.wasp.yes) why = "WASP";
    else if (P.var.yes) why = "variation";
    else if (P.outFilterBySJoutStage != 0) why = "outFilterType BySJout";
    else if (P.outSAMunmapped.within) why = "outSAMunmapped Within";
    else if (P.outReadsUnmapped != "None") why = "outReadsUnmapped";
    else if (P.scoreDelBase > 0 || P.scoreInsBase > 0) why = "positive indel base score";
    if (!why.empty()) {
        P.inOut->logMain << "SHORTEXIT: off (" << why << ")\n";
        return nullptr;
    }
    ShortExit *s = new ShortExit;
    s->mode = mode;
    intScore motif = std::max({(intScore) 0, P.scoreGapNoncan, P.scoreGapGCAG, P.scoreGapATAC});
    s->bonusUp = std::max({(intScore) P.pGe.sjdbScore, (intScore) (P.scoreGap + motif), (intScore) (P.scoreDelOpen + P.scoreDelBase)});
    s->costIns = P.scoreInsOpen + P.scoreInsBase;
    P.inOut->logMain << "SHORTEXIT: " << (mode == 1 ? "on" : "verify") << ", junction/deletion bonus <= " << s->bonusUp
                     << ", insertion cost <= " << s->costIns << "\n";
    return s;
}

std::string ShortExit::stats() const {
    std::ostringstream o;
    o << "SHORTEXIT: bounded=" << nBound << " proven=" << nProven << " windowsSkipped=" << nWinSkipped
      << " verified=" << nVerified << " verifyFail=" << nVerifyFail;
    return o.str();
}

// true if every window's transcripts are bounded below the "too short" thresholds
bool ReadAlign::shortExitBound(char **R, uint Lread) {
    const intScore thrS = std::max(P.outFilterScoreMin, (intScore) (P.outFilterScoreMinOverLread*(Lread-1)));
    const uint thrM = std::max(P.outFilterMatchNmin, (uint) (P.outFilterMatchNminOverLread*(Lread-1)));
    const intScore NEG = -(1 << 28);
    const intScore bonusUp = g_shortExit->bonusUp, costIns = g_shortExit->costIns;
    const char *G = mapGen.G;
    const int64 nG = (int64) mapGen.nGenome;

    //a seed is an exact match: a window holding a seed at least as long as both thresholds cannot be bounded below them
    for (uint iW=0; iW<nW; iW++) {
        for (uint ia=0; ia<nWA[iW]; ia++) {
            uint64 l = WA[iW][ia][WA_Length];
            if (l >= thrM && (intScore) l >= thrS)
                return false;
        };
    };

    for (uint iW=0; iW<nW; iW++) {
        if (nWA[iW]==0) continue;
        //distinct seed diagonals, ascending
        shortExitDiag.clear();
        for (uint ia=0; ia<nWA[iW]; ia++)
            shortExitDiag.push_back((int64) WA[iW][ia][WA_gStart] - (int64) WA[iW][ia][WA_rStart]);
        std::sort(shortExitDiag.begin(), shortExitDiag.end());
        shortExitDiag.erase(std::unique(shortExitDiag.begin(), shortExitDiag.end()), shortExitDiag.end());
        const uint K = shortExitDiag.size();
        uint roStr = revertStrand ? 1-WC[iW][WC_Str] : WC[iW][WC_Str];
        const char *Rs = R[roStr==0 ? 0 : 2];

        //per diagonal and read position: +1 match, -1 mismatch, 0 N or outside the genome
        shortExitM.resize((size_t) K*Lread);
        uint nMatchBound = 0;
        for (uint k=0; k<K; k++) {
            int8_t *m = shortExitM.data() + (size_t) k*Lread;
            int64 d = shortExitDiag[k];
            for (uint i=0; i<Lread; i++) {
                int64 g = d + (int64) i;
                char r = Rs[i];
                if (g<0 || g>=nG || r>3 || G[g]>3) m[i] = 0;
                else m[i] = (r==G[g]) ? 1 : -1;
            };
        };
        for (uint i=0; i<Lread; i++) {
            for (uint k=0; k<K; k++) {
                if (shortExitM[(size_t) k*Lread+i]==1) {nMatchBound++; break;};
            };
        };
        if (nMatchBound < thrM)
            continue; //every transcript of this window has too few matches

        //score bound: best chain over the diagonals
        shortExitS.assign(K, NEG);
        shortExitPrev.assign(K, NEG);
        intScore pref1 = NEG, pref2 = NEG; //max chain score ending at <= i-1, <= i-2
        for (uint i=0; i<Lread; i++) {
            std::swap(shortExitS, shortExitPrev);
            intScore runLower = NEG; //max of prev over strictly smaller diagonals
            intScore ins = pref2 > NEG ? pref2 + costIns : NEG;
            intScore rowMax = NEG;
            for (uint k=0; k<K; k++) {
                intScore base = 0; //start a chain here
                intScore p = shortExitPrev[k];
                if (p > base) base = p;
                if (runLower > NEG && runLower + bonusUp > base) base = runLower + bonusUp;
                if (ins > base) base = ins;
                intScore s = base + shortExitM[(size_t) k*Lread+i];
                shortExitS[k] = s;
                if (s > rowMax) rowMax = s;
                if (p > runLower) runLower = p;
            };
            pref2 = pref1;
            if (rowMax > pref1) pref1 = rowMax;
        };
        if (pref1 < thrS)
            continue; //every transcript of this window scores too low
        return false;
    };
    return true;
}
