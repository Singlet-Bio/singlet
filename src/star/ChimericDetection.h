#ifndef CODE_ChimericDetection
#define CODE_ChimericDetection

#include "IncludeDefine.h"
#include "Parameters.h"
#include "Transcript.h"
#include "ChimericAlign.h"
#include "Genome.h"

class ReadAlign;

class ChimericDetection {
    private:
        Parameters &P;
        ReadAlign *RA;
        Transcript ***trAll;
        uint nW, *nWinTr;
        char** Read1;
        Genome &outGen;

        // SINGLET_CHIM_PAIRPRUNE: the pair loop in two forms (see ChimericDetection_chimericDetectionMult.cpp)
        void pairLoopOld(uint nW, int &minScoreToConsider, int &chimScoreBest, std::size_t &bestChimAlign);
        void pairLoopNew(uint nW, int &minScoreToConsider, int &chimScoreBest, std::size_t &bestChimAlign);
        std::vector<ChimericSegment> flSeg; //segments of the transcripts that pass segmentCheck, in loop order (pairLoopNew)
        std::vector<int> flScore;           //align.maxScore of flSeg[i]
        std::vector<int> flSufMax;          //flSufMax[i] = max of flScore[j], j>i (defined for i < size-1)

    public:
        vector <ChimericAlign> chimAligns;

        ChimericDetection(Parameters &Pin, Transcript ***trAll, uint *nWinTr, char** Read1in, Genome &genomeIn, fstream *ostreamChimJunctionIn, ReadAlign *RA);
        bool chimericDetectionMult(uint nWin, uint *readLength, int maxNonChimAlignScore, ReadAlign *PEunmergedRA);
        fstream *ostreamChimJunction;
};

#endif
