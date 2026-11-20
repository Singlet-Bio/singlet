//#include "blocksOverlap.h"
#include "ChimericDetection.h"
#include "ChimericSegment.h"
#include "ReadAlign.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int chimericAlignScore (ChimericSegment & seg1, ChimericSegment & seg2)
{
    int chimScore=0;
    uint chimOverlap = seg2.roS>seg1.roS ?  (seg2.roS>seg1.roE ? 0 : seg1.roE-seg2.roS+1) : (seg2.roE<seg1.roS ? 0 : seg2.roE-seg1.roS+1);
    bool diffMates=(seg1.roE < seg1.align.readLength[0] && seg2.roS >= seg1.align.readLength[0]) || (seg2.roE < seg1.align.readLength[0] && seg1.roS >= seg1.align.readLength[0]);

    //segment lengths && (different mates || small gap between segments)
    if (seg1.roE > seg1.P.pCh.segmentMin + seg1.roS + chimOverlap && seg2.roE > seg1.P.pCh.segmentMin + seg2.roS + chimOverlap  \
        && ( diffMates || ( (seg1.roE + seg1.P.pCh.segmentReadGapMax + 1) >= seg2.roS && (seg2.roE + seg1.P.pCh.segmentReadGapMax + 1) >= seg1.roS ) ) )
    {
       chimScore = seg1.align.maxScore + seg2.align.maxScore - (int)chimOverlap; //subtract overlap to avoid double counting
    };

    return chimScore;
};

void ChimericDetection::pairLoopOld(uint nW, int &minScoreToConsider, int &chimScoreBest, std::size_t &bestChimAlign) {
//the pair loop as it was before SINGLET_CHIM_PAIRPRUNE (run with SINGLET_CHIM_PAIRPRUNE=0 and as the reference of =verify)

    for (uint iW1=0; iW1<nW; iW1++) {//cycle windows
        for (uint iA1=0; iA1<nWinTr[iW1]; iA1++) {//cycle aligns in the window

            ChimericSegment seg1(P,*trAll[iW1][iA1]);

            if (!seg1.segmentCheck())
                continue; //seg1 is bad - skip

            for (uint iW2=iW1; iW2<nW; iW2++) {//check all windows for chimeras
                for (uint iA2=(iW1!=iW2 ? 0 : iA1+1); iA2<nWinTr[iW2]; iA2++) {//cycle over aligns in the window
                    //for the same window, start iA2 at iA1+1 to avoid duplicating

                    ChimericSegment seg2(P,*trAll[iW2][iA2]);

                    if (!seg2.segmentCheck())
                        continue; //seg2 is bad - skip

                    if (seg1.str!=0 && seg2.str!=0 && seg2.str!=seg1.str)
                        continue; //chimeric segments have to have consistent strands. TODO: make this optional

                    int chimScore=chimericAlignScore(seg1,seg2);

                    if (chimScore >= minScoreToConsider) {//candidate chimera
                        ChimericAlign chAl(seg1, seg2, chimScore, outGen, RA);

                        if (!chAl.chimericCheck())
                            continue; //check chimeric alignment

                        //re-calculated chimScoreBest includes non-canonical penalty, so the re-calculated score is lower, in some cases it goes to 0 if some checks are not passed
                        chAl.chimericStitching(outGen.G, Read1);
                        // rescore after stitching.
                        if (chAl.chimScore >= minScoreToConsider) { // survived stitching.
                            chimAligns.push_back(chAl);//add this chimeric alignment

                            if (chimAligns.back().chimScore > chimScoreBest) {
                                chimScoreBest=chimAligns.back().chimScore;
                                bestChimAlign = chimAligns.size()-1;
                                if ((chimScoreBest - (int)P.pCh.multimapScoreRange) > minScoreToConsider)
                                    // best score increased, so subsequent alignment candidates must score higher
                                    minScoreToConsider = chimScoreBest - (int)P.pCh.multimapScoreRange;
                            };
                        } // endif stitched chimera survived.
                        else {
                            // al1, al2 allocated during stitching
                            delete chAl.al1;
                            delete chAl.al2;
                        };

                    }; // endif meets chim score criteria
                };//cycle over window2 aligns
            };//cycle over window2
        };//cycle over window1 aligns
    };//cycle over window1
};

//SINGLET_CHIM_PAIRPRUNE: 0 = old loop, 1 = pruned loop (default), verify = both, compared per read. Read once per process.
static int chimPairPruneMode() {
    const char *e = getenv("SINGLET_CHIM_PAIRPRUNE");
    if (e == NULL || e[0] == 0)
        return 1;
    if (strcmp(e, "0") == 0)
        return 0;
    if (strcmp(e, "verify") == 0)
        return 2;
    return 1;
};

void ChimericDetection::pairLoopNew(uint nW, int &minScoreToConsider, int &chimScoreBest, std::size_t &bestChimAlign) {
    //Flatten once per read, in the loop order of pairLoopOld (windows, then aligns in the window). Each ChimericSegment is built and
    //checked once. Transcripts that fail segmentCheck() are left out: the old loop skips them both as seg1 and as seg2, so the pairs that
    //remain, and their order, are exactly the pairs the old loop scores. The segment constructor and segmentCheck() read only
    //P.pCh.segmentMin and the Transcript (intronMotifs, Str, exons, Lread, readLength, rLength) and write only the segment itself;
    //nothing in the loop writes to trAll[][] (chimericStitching works on copies), so building them before the loop changes no value.
    //The segments stay in flSeg (reserved, never reallocated) until this function returns; a ChimericAlign holds copies of the two
    //segments (members seg1, seg2, whose `align` is a reference to the Transcript in trAll) and al1/al2 point into trAll or to
    //copies it allocates itself, so it does not depend on where the segments live, only on trAll, as in the old loop.
    flSeg.clear();
    flScore.clear();
    std::size_t nTot = 0;
    for (uint iW1=0; iW1<nW; iW1++)
        nTot += nWinTr[iW1];
    flSeg.reserve(nTot);
    flScore.reserve(nTot);
    for (uint iW1=0; iW1<nW; iW1++) {
        for (uint iA1=0; iA1<nWinTr[iW1]; iA1++) {
            flSeg.emplace_back(P,*trAll[iW1][iA1]);
            if (flSeg.back().segmentCheck()) {
                flScore.push_back(flSeg.back().align.maxScore);
            } else {
                flSeg.pop_back();
            };
        };
    };

    const std::size_t nF = flSeg.size();
    flSufMax.assign(nF, 0);
    if (nF >= 2) {
        int m = flScore[nF-1];
        for (std::size_t k = nF-1; k-- > 0; ) {
            flSufMax[k] = m;
            if (flScore[k] > m)
                m = flScore[k];
        };
    };

    for (std::size_t i = 0; i+1 < nF; i++) {//seg1; i+1<nF: the last segment has no later partner
        const int score1 = flScore[i];

        //Exactness of both bounds below. chimericAlignScore returns either 0 or maxScore1 + maxScore2 - (int)chimOverlap. chimOverlap is
        //unsigned and by its definition (line 9) is either 0 or (end - start + 1) of two values with start <= end in the branch taken, so
        //it is in [0, max(roE)+1], bounded by the read length (at most DEF_readSeqLengthMax = 650 bases per read, far below INT_MAX);
        //therefore (int)chimOverlap >= 0 and chimScore <= score1 + score2 whenever it is not 0. A pair acts only if
        //chimScore >= minScoreToConsider (nothing else in the loop has a side effect), so with minScoreToConsider > 0 a pair with
        //score1 + score2 < minScoreToConsider has chimScore == 0 or < minScoreToConsider: it cannot act. The condition
        //minScoreToConsider > 0 is needed: with a threshold <= 0 a pair with score 0 is a candidate in the old loop.
        //minScoreToConsider only increases inside the loop, so a pair excluded by the value at the time of the test is also excluded by
        //every later value, and the test is made with the current value for every pair, as the old loop does.
        //Second bound: no later partner can have a score above flSufMax[i] (maximum over the later segments that pass segmentCheck).
        if (minScoreToConsider > 0 && score1 + flSufMax[i] < minScoreToConsider)
            continue;

        ChimericSegment &seg1 = flSeg[i];

        for (std::size_t j = i+1; j < nF; j++) {//seg2: the later pairs in the old order (same window: iA2>iA1; later windows)
            if (minScoreToConsider > 0 && score1 + flScore[j] < minScoreToConsider)
                continue; //bound above: cannot be a candidate

            ChimericSegment &seg2 = flSeg[j];

            if (seg1.str!=0 && seg2.str!=0 && seg2.str!=seg1.str)
                continue; //chimeric segments have to have consistent strands. TODO: make this optional

            int chimScore=chimericAlignScore(seg1,seg2);

            if (chimScore >= minScoreToConsider) {//candidate chimera, the block below is the old loop's, unchanged
                ChimericAlign chAl(seg1, seg2, chimScore, outGen, RA);

                if (!chAl.chimericCheck())
                    continue; //check chimeric alignment

                chAl.chimericStitching(outGen.G, Read1);
                if (chAl.chimScore >= minScoreToConsider) { // survived stitching.
                    chimAligns.push_back(chAl);//add this chimeric alignment

                    if (chimAligns.back().chimScore > chimScoreBest) {
                        chimScoreBest=chimAligns.back().chimScore;
                        bestChimAlign = chimAligns.size()-1;
                        if ((chimScoreBest - (int)P.pCh.multimapScoreRange) > minScoreToConsider)
                            // best score increased, so subsequent alignment candidates must score higher
                            minScoreToConsider = chimScoreBest - (int)P.pCh.multimapScoreRange;
                    };
                } // endif stitched chimera survived.
                else {
                    // al1, al2 allocated during stitching
                    delete chAl.al1;
                    delete chAl.al2;
                };
            };
        };
    };
};

/////////////////////////////////////////////////////////////
bool ChimericDetection::chimericDetectionMult(uint nW, uint *readLength, int maxNonChimAlignScore, ReadAlign *PEunmergedRA) {

//     for (uint ii=0;ii<chimAligns.size();ii++) {//deallocate aligns
//         if (chimAligns.at(ii).stitchingDone) {//al1,al2 were allocated
//             delete chimAligns.at(ii).al1;
//             delete chimAligns.at(ii).al2;
//         };
//     };

    for (auto cAit=chimAligns.begin(); cAit<chimAligns.end(); cAit++) {//deallocate aligns
        if (cAit->stitchingDone) {//al1,al2 were allocated
            delete cAit->al1;
            delete cAit->al2;
        };
    };

    chimAligns.clear();
    int chimScoreBest=0;
    std::size_t bestChimAlign=0; // points to element of chimAligns with highest chimScoreBest

    int maxPossibleAlignScore = (int)(readLength[0]+readLength[1]);
    int minScoreToConsider = P.pCh.scoreMin;
    if (maxNonChimAlignScore >= minScoreToConsider)
        minScoreToConsider = maxNonChimAlignScore + 1;
    if ((maxPossibleAlignScore - P.pCh.scoreDropMax) > minScoreToConsider)
        minScoreToConsider = maxPossibleAlignScore - P.pCh.scoreDropMax;

    static const int pairPruneMode = chimPairPruneMode(); //read once; 0 old loop, 1 pruned loop (default), 2 both and compare

    if (pairPruneMode == 0) {
        pairLoopOld(nW, minScoreToConsider, chimScoreBest, bestChimAlign);
    } else if (pairPruneMode == 1) {
        pairLoopNew(nW, minScoreToConsider, chimScoreBest, bestChimAlign);
    } else {//verify: the old loop into oldAligns, the new loop into chimAligns, compare, continue with the new result
        int minOld = minScoreToConsider, bestOld = 0;
        std::size_t bestIdxOld = 0;
        pairLoopOld(nW, minOld, bestOld, bestIdxOld);
        std::vector<ChimericAlign> oldAligns;
        oldAligns.swap(chimAligns); //oldAligns owns the al1/al2 allocated by the old loop (stitchingDone)
        pairLoopNew(nW, minScoreToConsider, chimScoreBest, bestChimAlign);
        std::size_t firstDiff = chimAligns.size() == oldAligns.size() ? (std::size_t)-1 : std::min(chimAligns.size(), oldAligns.size());
        for (std::size_t i = 0; i < chimAligns.size() && i < oldAligns.size() && firstDiff == (std::size_t)-1; i++) {
            const ChimericAlign &o = oldAligns[i];
            const ChimericAlign &n = chimAligns[i];
            if (&o.seg1.align != &n.seg1.align || &o.seg2.align != &n.seg2.align || o.seg1.str != n.seg1.str || o.seg2.str != n.seg2.str \
                || o.chimScore != n.chimScore || o.stitchingDone != n.stitchingDone || o.ex1 != n.ex1 || o.ex2 != n.ex2 \
                || o.chimJ1 != n.chimJ1 || o.chimJ2 != n.chimJ2 || o.chimRepeat1 != n.chimRepeat1 || o.chimRepeat2 != n.chimRepeat2 \
                || o.chimMotif != n.chimMotif || o.chimStr != n.chimStr || o.al1->maxScore != n.al1->maxScore || o.al2->maxScore != n.al2->maxScore \
                || o.al1->nExons != n.al1->nExons || o.al2->nExons != n.al2->nExons)
                firstDiff = i;
        };
        if (firstDiff != (std::size_t)-1 || chimScoreBest != bestOld || bestChimAlign != bestIdxOld || minScoreToConsider != minOld) {
            fprintf(stderr, "[singlet] SINGLET_CHIM_PAIRPRUNE=verify MISMATCH: chimAligns old %zu new %zu first difference %lld; chimScoreBest old %d new %d; bestChimAlign old %zu new %zu; minScoreToConsider old %d new %d; nW %llu\n",
                    oldAligns.size(), chimAligns.size(), firstDiff == (std::size_t)-1 ? -1LL : (long long)firstDiff, bestOld, chimScoreBest, bestIdxOld, bestChimAlign, minOld, minScoreToConsider, (unsigned long long)nW);
            fflush(stderr);
            std::_Exit(1);
        };
        for (auto cAit=oldAligns.begin(); cAit<oldAligns.end(); cAit++) {//free the old loop's allocations (the new loop's stay in chimAligns)
            if (cAit->stitchingDone) {
                delete cAit->al1;
                delete cAit->al2;
            };
        };
    };

    if (chimScoreBest==0)
        return false;

    uint chimN=0;
    for (auto cAit=chimAligns.begin(); cAit<chimAligns.end(); cAit++) {
        //scan all chimeras, find the number within score range
        if (cAit->chimScore >= minScoreToConsider)
            ++chimN;
    };

    if (chimN > P.pCh.multimapNmax) //too many loci
        return false;

    uint iTr = 0; //keep track of "HI" SAM attribute
    for (std::size_t i = 0; i < chimAligns.size(); i++) {//output chimeras within score range
        if (chimAligns[i].chimScore >= minScoreToConsider) {

            if (P.pCh.out.junctions)
                chimAligns[i].chimericJunctionOutput(*ostreamChimJunction, chimN, maxNonChimAlignScore, PEunmergedRA != NULL, chimScoreBest, maxPossibleAlignScore);

            if (P.pCh.out.bam) {
                // convert merged SE chimera to PE chimera if this is a merged chimera
                if (PEunmergedRA != NULL) {
                    chimAligns[i].RA = PEunmergedRA;
                    chimAligns[i].RA->peOverlapChimericSEtoPE(chimAligns[i].al1, chimAligns[i].al2, chimAligns[i].al1, chimAligns[i].al2);
                };
                chimAligns[i].chimericBAMoutput(chimAligns[i].al1, chimAligns[i].al2, chimAligns[i].RA, iTr, chimN, i == bestChimAlign, P);
            };
            iTr++;

        };
    };

    return chimN > 0;
};//END
