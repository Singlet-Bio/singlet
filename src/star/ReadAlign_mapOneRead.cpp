#include "ReadAlign.h"
#include "SequenceFuns.h"
#include "Stats.h"
#include "serviceFuns.cpp"

int ReadAlign::mapOneRead() {

    #ifdef OFF_BEFORE_SEEDING
        #warning OFF_BEFORE_SEEDING
        nW=0;
        return 0;
    #endif

    revertStrand = false; //the 2nd read is awlays on opposite strand. 1st and 2nd reads have already been reversed.

    if (Lread>0) {
        Nsplit=qualitySplit(Read1[0], Lread, P.maxNsplit, P.seedSplitMin, splitR);
        // splitR[0][fragnum] => good region start position   (from SequenceFuns.cpp)
        // splitR[1][fragnum] => good reagion length
        // splitR[2][fragnum] => fragnum ?
    } else {
        Nsplit=0;
    };

    resetN(); //reset aligns counters to 0

    //reset/initialize a transcript
    trInit->reset();
    trInit->Chr=0;    trInit->Str=0; trInit->roStr=0;    trInit->cStart=0;     trInit->gLength=0; //to generate nice output of 0 for non-mapped reads
    trInit->iRead=iRead;
    trInit->Lread=Lread;
    trInit->nExons=0;
    trInit->readLengthOriginal=readLengthOriginal;
    trInit->readLengthPairOriginal=readLengthPairOriginal;
    trInit->readLength=readLength;
    trInit->readNmates=readNmates; //not readNends: this is alignment
    trInit->readName=readName;

    trBest=trInit;

    uint seedSearchStartLmax=min(P.seedSearchStartLmax, // 50
                                  (uint) (P.seedSearchStartLmaxOverLread*(Lread-1))); // read length
    // align all good pieces
    for (uint ip=0; ip<Nsplit; ip++) {


        // if the good piece is long, then try multiple times to map it
        uint Nstart = P.seedSearchStartLmax>0 && seedSearchStartLmax<splitR[1][ip] ? splitR[1][ip]/seedSearchStartLmax+1 : 1;
        uint Lstart = splitR[1][ip]/Nstart;  // length of segment to map.
        bool flagDirMap=true;

        for (uint iDir=0; iDir<2; iDir++) {//loop over two directions (leftToRight and RightToLeft)

            uint Lmapped, L;

            for (uint istart=0; istart<Nstart; istart++) {  // each try to map a segment.

//               #ifdef COMPILE_FOR_LONG_READS
//
//               #else
                if (flagDirMap || istart>0) {//check if the 1st piece in reveree direction does not need to be remapped
                    Lmapped=0;  // length of segment mapped so far.

                    // begin mapping starting from segment start position (istart*Lstart)
                    while ( istart*Lstart + Lmapped + P.seedMapMin < splitR[1][ip] ) {//map until unmapped portion is <=minLmap (default: 5)

                        // Shift is the position in the read to begin mapping from.
                        uint Shift = iDir==0 ? ( splitR[0][ip] + istart*Lstart + Lmapped ) : \
                                   ( splitR[0][ip] + splitR[1][ip] - istart*Lstart-1-Lmapped); //choose Shift for forward or reverse

                        //uint seedLength=min(splitR[1][ip] - Lmapped - istart*Lstart, P.seedSearchLmax);
                        uint seedLength=splitR[1][ip] - Lmapped - istart*Lstart; // what's left of the read to align.

#ifdef SA_NEXT_SEED_PREFETCH
                        // Speculatively prefetch SA bounds for the NEXT seed before processing the CURRENT seed.
                        // Assumes this seed maps L=seedLength bases (true for most uniquely-mapped scRNA-seq reads).
                        // The prefetch fires ~full-maxMappableLength cycles before the next SA access (~1000-5000 cycles)
                        // — enough to hide DRAM/L3 miss latency — vs SA_BOUNDARY_PREFETCH's ~80-instruction window.
                        {
                            uint nxt_Lmapped = Lmapped + seedLength; // speculative
                            if (istart * Lstart + nxt_Lmapped + P.seedMapMin < splitR[1][ip]) {
                                uint nxt_Shift = iDir == 0
                                    ? (splitR[0][ip] + istart * Lstart + nxt_Lmapped)
                                    : (splitR[0][ip] + splitR[1][ip] - istart * Lstart - 1 - nxt_Lmapped);
                                uint nxt_seedLen = splitR[1][ip] - nxt_Lmapped - istart * Lstart;
                                uint nxt_Lmax = min((uint)P.pGe.gSAindexNbases, nxt_seedLen);
                                uint nxt_ind1 = 0;
                                if (iDir == 0) {
                                    for (uint ii = 0; ii < nxt_Lmax; ii++) {
                                        nxt_ind1 <<= 2LLU;
                                        nxt_ind1 += (uint)Read1[0][nxt_Shift + ii];
                                    }
                                } else {
                                    for (uint ii = 0; ii < nxt_Lmax; ii++) {
                                        nxt_ind1 <<= 2LLU;
                                        nxt_ind1 += 3u - (uint)Read1[0][nxt_Shift - ii];
                                    }
                                }
                                uint nxt_Lind = nxt_Lmax;
                                uint nxt_iSA1 = 0;
                                while (nxt_Lind > 0) {
                                    nxt_iSA1 = mapGen.SAi[mapGen.genomeSAindexStart[nxt_Lind - 1] + nxt_ind1];
                                    if ((nxt_iSA1 & mapGen.SAiMarkAbsentMaskC) == 0) break;
                                    --nxt_Lind;
                                    nxt_ind1 >>= 2;
                                }
                                if (nxt_Lind > 0) {
                                    uint nxt_iSA2 = mapGen.nSA - 1;
                                    if (mapGen.genomeSAindexStart[nxt_Lind-1] + nxt_ind1 + 1 < mapGen.genomeSAindexStart[nxt_Lind]) {
                                        nxt_iSA2 = mapGen.SAi[mapGen.genomeSAindexStart[nxt_Lind-1] + nxt_ind1 + 1];
                                        if ((nxt_iSA2 & mapGen.SAiMarkAbsentMaskC) != 0)
                                            nxt_iSA2 = mapGen.nSA - 1;
                                        else
                                            nxt_iSA2 = (nxt_iSA2 & mapGen.SAiMarkNmask) - 1;
                                    }
                                    const uint64_t nxt_wl = mapGen.SA.wordLength;
                                    char* const nxt_sa = mapGen.SA.charArray;
                                    uint nxt_i1 = nxt_iSA1 & mapGen.SAiMarkNmask;
                                    __builtin_prefetch(nxt_sa + (uint64_t)nxt_i1   * nxt_wl / 8, 0, 0);
                                    __builtin_prefetch(nxt_sa + (uint64_t)nxt_iSA2 * nxt_wl / 8, 0, 0);
                                    if (nxt_i1 < nxt_iSA2)
                                        __builtin_prefetch(nxt_sa + (uint64_t)((nxt_i1 + nxt_iSA2) / 2) * nxt_wl / 8, 0, 0);
                                }
                            }
                        }
#endif
                        maxMappableLength2strands(Shift, seedLength, iDir, 0, mapGen.nSA-1, L, splitR[2][ip]);//L=max mappable length, unique or multiple
                        if (iDir==0 && istart==0 && Lmapped==0 && Shift+L == splitR[1][ip] ) {//this piece maps full length and does not need to be mapped from the opposite direction
                            flagDirMap=false;
                        };
                        Lmapped+=L;
                    };//while ( istart*Lstart + Lmapped + P.minLmap < splitR[1][ip] )
                };//if (flagDirMap || istart>0)

                if (P.seedSearchLmax>0) {//search fixed length. Not very efficient, need to improve
                    // off by default.
                    uint Shift = iDir==0 ? ( splitR[0][ip] + istart*Lstart ) : \
                                   ( splitR[0][ip] + splitR[1][ip] - istart*Lstart-1); //choose Shift for forward or reverse
                    uint seedLength = min(P.seedSearchLmax, iDir==0 ? (splitR[0][ip] + splitR[1][ip]-Shift):(Shift+1) );
                    maxMappableLength2strands(Shift, seedLength, iDir, 0, mapGen.nSA-1, L, splitR[2][ip]);//L=max mappable length, unique or multiple
                };


//               #endif
            };//for (uint istart=0; istart<Nstart; istart++)
        };
    };

    #ifdef OFF_AFTER_SEEDING
        #warning OFF_AFTER_SEEDING
        return 0;
    #endif

    if (Lread<P.outFilterMatchNmin) {//read is too short (trimmed too much?)
        mapMarker=MARKER_READ_TOO_SHORT;
        trBest->rLength=0; //min good piece length
        nW=0;
    } else if (Nsplit==0) {//no good pieces
        mapMarker=MARKER_NO_GOOD_PIECES;
        trBest->rLength=splitR[1][0]; //min good piece length
        nW=0;
    } else if (Nsplit>0 && nA==0) {
        mapMarker=MARKER_ALL_PIECES_EXCEED_seedMultimapNmax;
        trBest->rLength=multNminL;
        nW=0;
    } else if (Nsplit>0 && nA>0) {//otherwise there are no good pieces, or all pieces map too many times: read cannot be mapped
//         qsort((void*) PC, nP, sizeof(uint)*PC_SIZE, funCompareUint2);//sort PC by rStart and length
        stitchPieces(Read1, Lread);
    };

    return 0;
};
