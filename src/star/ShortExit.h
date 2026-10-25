#ifndef H_ShortExit
#define H_ShortExit

// Exact early exit for reads that STAR reports as "unmapped: too short".
//
// mappedFilter() calls a read too short when its best transcript has maxScore below
// max(outFilterScoreMin, outFilterScoreMinOverLread*(Lread-1)) or nMatch below
// max(outFilterMatchNmin, outFilterMatchNminOverLread*(Lread-1)). Every base of a stitched transcript lies on the
// diagonal (genome start - read start) of one of the seeds assigned to its window: seeds are exact matches, gaps
// between two seeds are filled on one of the two diagonals, and extensions continue the first/last seed's diagonal.
// So, after the seeds are assigned to windows and before stitching, two bounds hold for every transcript of a window:
//   nMatch   <= number of read positions that match the genome on at least one of the window's seed diagonals;
//   maxScore <= best chain of read positions over those diagonals scored +1 match / -1 mismatch / 0 N, where moving
//               to a larger diagonal (junction or deletion) earns at most max(sjdbScore, scoreGap + best motif
//               penalty, scoreDelOpen + scoreDelBase) and skipping read bases (insertion) costs at least
//               scoreInsOpen + scoreInsBase; STAR's genomic-length term and max(0, .) only lower it further.
// If every window is bounded below the thresholds, STAR's output is fixed: "too short" if some window yields a
// transcript with maxScore > 0 (then nW > 0), otherwise "other" (nW = 0). Windows are then stitched in STAR's own order
// only until the first one with a positive best transcript; the remaining windows cannot change the outcome, and an
// unmapped read's records (no BAM line with outSAMunmapped None, nTr = 0 to Solo, unmappedShort in Log.final.out)
// do not depend on which transcript was best.
//
// SINGLET_STAR_SHORTEXIT=0 (default) off, 1 on, verify: compute the bound, stitch every window as before and count
// reads where a proven bound did not end as "too short" (must be 0).

#include "IncludeDefine.h"
#include <atomic>
#include <string>

class Parameters;

class ShortExit {
public:
    static ShortExit *create(Parameters &P); //nullptr if off or not exact for these parameters
    int mode; //1 on, 2 verify
    intScore bonusUp, costIns; //score bounds for a move to a larger diagonal / an insertion
    std::atomic<uint64> nBound{0}, nProven{0}, nWinSkipped{0}, nVerified{0}, nVerifyFail{0};
    std::string stats() const;
};

extern ShortExit *g_shortExit;

#endif
