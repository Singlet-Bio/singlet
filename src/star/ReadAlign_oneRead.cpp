#include "StarProfile.h"
#include "ReadAlign.h"
#include "readLoad.h"
#include "readBarcodeLoad.h"
#include "SequenceFuns.h"
#include "ErrorWarning.h"
#include "GlobalVariables.h"

#include "SeqCache.h"

int ReadAlign::oneRead() {//process one read: load, map, write

    //load read name, sequence, quality from the streams into internal arrays
    int readStatus[P.readNends];

    PROF_T0(tLoad);
    for (uint32 im=0; im<P.readNends; im++) {
        readStatus[im] = readLoad(*(readInStream[im]), P, readLength[im], readLengthOriginal[im], readNameMates[im], Read0[im], Read1[im], Qual0[im], clipMates[im], iReadAll, readFilesIndex, readFilter, readNameExtra[im]);
        if (readStatus[im] != readStatus[0]) {//check if the end of file was reached or not for all files
            ostringstream errOut;
            errOut << "EXITING because of FATAL ERROR: read files are not consistent, reached the end of the one before the other one\n";
            errOut << "SOLUTION: Check you your input files: they may be corrupted\n";
            exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        };
    };

    PROF_ADD(PROF_readLoad, tLoad);
    if (readStatus[0]==-1) {//finished with the stream
        return -1;
    };    
    
    if (P.outFilterBySJoutStage != 2) {
        for (uint32 im=0; im<P.readNmates; im++) {//not readNends: the barcode quality will be calculated separately
            for (uint64 ix=clipMates[im][0].clippedN; ix<readLengthOriginal[im]-clipMates[im][1].clippedN; ix++) {
                qualHist[im][(uint8)Qual0[im][ix]]++;
            };
        };
    };
    
    if (P.readNmates==2) {//combine two mates together
        Lread=readLength[0]+readLength[1]+1;
        readLengthPairOriginal=readLengthOriginal[0]+readLengthOriginal[1]+1;
        if (Lread>DEF_readSeqLengthMax) {
            ostringstream errOut;
            errOut << "EXITING because of FATAL ERROR in reads input: Lread of the pair = " << Lread << "   while DEF_readSeqLengthMax=" << DEF_readSeqLengthMax <<endl;
            errOut << "Read Name="<<readNameMates[0]<<endl;
            errOut << "SOLUTION: increase DEF_readSeqLengthMax in IncludeDefine.h and re-compile STAR"<<endl<<flush;
            exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        };

        //marker for spacer base
        Read1[0][readLength[0]]=MARK_FRAG_SPACER_BASE;
        
        //copy 2nd mate into Read1[0] & reverse-complement
        complementSeqNumbers(Read1[1],Read1[0]+readLength[0]+1,readLength[1]);//complement. Here Read1[1] is still the 2nd mate's numeric-sequence. Later Read1[1] will be reverse complement of the combined read.
        for (uint ii=0;ii<readLength[1]/2;ii++) {
            swap(Read1[0][Lread-ii-1],Read1[0][ii+readLength[0]+1]); //reverse
        };

    } else {//1 mate

        if (readStatus[0]==-1) {//finished with the stream
            return -1;
        };

        Lread=readLength[0];
        readLengthPairOriginal=readLengthOriginal[0];
        readLength[1]=0;

    };
      
    readFileType=readStatus[0];

    complementSeqNumbers(Read1[0],Read1[1],Lread); //returns complement of Reads[ii]
    for (uint ii=0;ii<Lread;ii++) {//reverse
        Read1[2][Lread-ii-1]=Read1[1][ii];
    };

    statsRA.readN++;
    statsRA.readBases += readLength[0]+readLength[1];

    //max number of mismatches allowed for this read
    outFilterMismatchNmaxTotal=min(P.outFilterMismatchNmax, (uint) (P.outFilterMismatchNoverReadLmax*(readLength[0]+readLength[1])));

    //map the read, or take its alignment from the sequence cache (SeqCache.h)
    PROF_T0(tMap);
    bool fromCache = false;
    uint64 seqHash = 0;
    bool cacheable = g_seqCache != nullptr && seqCacheable();
    if (cacheable) {
        seqHash = SeqCache::hashSeq(Read1[0], Lread);
        if (g_seqCache->lookup(seqHash, Read1[0], Lread, seqCacheBlob))
            fromCache = !g_seqCache->verify;
    } else if (g_seqCache != nullptr) {
        g_seqCache->nUncacheable++;
    };

    if (fromCache) {
        seqCacheRestore(seqCacheBlob);
        mappedFilter();
        alignsGenOut.alBest = trBest; //as transformGenome() does when there is no transformation
    } else {
        if (P.pGe.gType==101) {//SpliceGraph
            mapOneReadSpliceGraph();
        } else {//all other cases - standard alignment algorithm
            mapOneRead();
        };

        peOverlapMergeMap();

        multMapSelect();

        mappedFilter();

        transformGenome();//for now genome transformation happens after multimapper selection, and mapping filter

        if (!peOv.yes) {//if the alignment was not mates merged - otherwise the chimeric detection was already done
            chimericDetection();
        };

        if (cacheable) {
            bool hit = !seqCacheBlob.empty(); //lookup() filled the blob: verify mode
            if (seqCacheEncode(seqCacheEncoded)) {
                if (hit) {
                    g_seqCache->nVerified++;
                    if (seqCacheEncoded != seqCacheBlob) {
                        if (g_seqCache->nVerifyFail++ < 20) {
                            ostringstream o;
                            o << "SEQCACHE verify mismatch: read " << readName << " iRead " << iReadAll << "\n  cached:";
                            for (auto c : seqCacheBlob) o << ' ' << (int) c;
                            o << "\n  mapped:";
                            for (auto c : seqCacheEncoded) o << ' ' << (int) c;
                            P.inOut->logMain << o.str() << "\n" << flush;
                        };
                    };
                } else {
                    g_seqCache->insert(seqHash, Read1[0], Lread, seqCacheEncoded.data(), seqCacheEncoded.size());
                };
            } else {
                g_seqCache->nUncacheable++;
            };
        };
    };
    seqCacheBlob.clear();
    PROF_ADD(PROF_map, tMap);
#ifdef STAR_PROFILE
    unsigned long long profMapCyc = __rdtsc() - tMap;
    int profUnique = (!fromCache && unmapType==-1 && nTr==1) ? 1 : 0;
    {
        int oc = fromCache ? 5 : unmapType==-1 ? (nTr==1 ? 0 : 1) : unmapType==3 ? 2 : unmapType==1 ? 3 : 4;
        g_profOutcome[oc][0] += __rdtsc() - tMap;
        g_profOutcome[oc][1]++;
    }
#endif

    if (P.pCh.out.bam && chimRecord) {//chimeric alignment was recorded in main BAM files, and it contains the representative portion, so non-chimeric aligmnent is not output
        return 0;
    };

    waspMap();

    #ifdef OFF_BEFORE_OUTPUT
        #warning OFF_BEFORE_OUTPUT
        return 0;
    #endif

    //write out alignments
    PROF_T0(tOut);
    outputAlignments();
    PROF_ADD(PROF_output, tOut);
#ifdef STAR_PROFILE
    if (profUnique) {//unique, mapped by STAR: class by gene concordance and alignment shape (txome-first ceiling)
        Transcript &t = *trBest;
        bool gene1 = readAnnot.annotFeatures[SoloFeatureTypes::Gene].fSet.size()==1;
        bool chrM = mapGen.chrName[t.Chr]=="chrM" || mapGen.chrName[t.Chr]=="MT";
        uint64 cov = 0;
        for (uint64 i=0; i<t.nExons; i++) cov += t.exons[i][EX_L];
        bool annotSJ = true;
        for (uint64 i=0; i+1<t.nExons; i++) if (t.canonSJ[i]<0 || t.sjAnnot[i]!=1) annotSJ = false;
        int cls;
        if (!gene1 || chrM || t.nDel>0 || t.nIns>0 || !annotSJ) cls = 6;
        else {
            int sub = (t.nMM==0 && cov==Lread) ? 0 : (t.nMM==1 && cov==Lread) ? 1 : 2;
            cls = sub*2 + (t.nExons>1 ? 1 : 0);
        };
        g_profClass[cls][0] += profMapCyc;
        g_profClass[cls][1]++;
    };
#endif

    {
    #ifdef DEBUG_OutputLastRead
        lastReadStream.seekp(ios::beg);
        lastReadStream << iReadAll <<" "<< readName <<endl;
    #endif
    };

    return 0;

};


