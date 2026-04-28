#include "ReadAlignChunk.h"
#include "GlobalVariables.h"
#include "ThreadControl.h"
#include "ErrorWarning.h"
#include SAMTOOLS_BGZF_H

#include <algorithm>
#include <vector>
#include <numeric>
#include <cstring>

// Sort reads within a chunk buffer by their 14-mer prefix.
// Consecutive reads sharing the same prefix will access the same SA index range,
// giving near-100% L2/L3 cache hit rate on the suffix array binary search
// (vs ~80% miss rate with random read order). See SuffixArrayFuns.cpp.
//
// Cost: ~O(n log n) sort + O(chunk_size) memcpy ≈ 0.1-0.2% of alignment time.
// Benefit: -4.0% overall (12-trial alternating A/B, paired t(11)=-3.72, p=0.003,
//          Cohen's d=1.24). Cold-cache: -6.3% (p=0.0005). Sort wins 11/12 trials.
#define OPT_SORT_CHUNK_BY_PREFIX

#ifdef OPT_SORT_CHUNK_BY_PREFIX
static void sortChunkByPrefix(char** chunkIn, std::array<uint64, MAX_N_MATES> &sizes, uint nEnds)
{
    if (sizes[0] == 0) return;

    // 1. Scan chunkIn[0] (biological read) to find record boundaries and extract 14-mer prefixes.
    //    Record format: @header\nSEQUENCE\n+\nQUALITY\n  (4 lines)
    struct Rec { uint64 off[2]; uint32 len[2]; uint32 key; };
    std::vector<Rec> recs;
    recs.reserve(sizes[0] / 200);

    uint64 pos[2] = {0, 0};
    while (pos[0] < sizes[0]) {
        Rec r;
        for (uint m = 0; m < nEnds; m++) {
            r.off[m] = pos[m];
            for (int line = 0; line < 4; line++) {
                char *buf = chunkIn[m];
                uint64 sz = sizes[m];
                if (m == 0 && line == 1) {
                    uint32 key = 0;
                    for (int b = 0; b < 14 && pos[m] + b < sz && buf[pos[m] + b] != '\n'; b++) {
                        key <<= 2;
                        char c = buf[pos[m] + b];
                        if (c == 'C' || c == 'c' || c == 1) key |= 1;
                        else if (c == 'G' || c == 'g' || c == 2) key |= 2;
                        else if (c == 'T' || c == 't' || c == 3) key |= 3;
                    }
                    r.key = key;
                }
                while (pos[m] < sz && buf[pos[m]] != '\n') pos[m]++;
                pos[m]++;
            }
            r.len[m] = (uint32)(pos[m] - r.off[m]);
        }
        for (uint m = nEnds; m < 2; m++) { r.off[m] = 0; r.len[m] = 0; }
        recs.push_back(r);
    }

    if (recs.size() < 2) return;

    std::vector<uint32> order(recs.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32 a, uint32 b) {
        return recs[a].key < recs[b].key;
    });

    bool sorted = true;
    for (uint32 i = 1; i < order.size(); i++) {
        if (order[i] < order[i-1]) { sorted = false; break; }
    }
    if (sorted) return;

    for (uint m = 0; m < nEnds; m++) {
        char *tmp = new char[sizes[m]];
        uint64 off = 0;
        for (uint32 idx : order) {
            const Rec &rec = recs[idx];
            std::memcpy(tmp + off, chunkIn[m] + rec.off[m], rec.len[m]);
            off += rec.len[m];
        }
        std::memcpy(chunkIn[m], tmp, sizes[m]);
        delete[] tmp;
    }
}
#endif

void ReadAlignChunk::mapChunk() {//map one chunk. Input reads stream has to be setup in RA->readInStream[ii]
    
    for (uint32 im=0; im<1; im++) {//hardcoded mate 1 5p onyl for now
        RA->clipMates[im][0].clipChunk(chunkIn[im], chunkInSizeBytesTotal[im]);
    };

    #ifdef OPT_SORT_CHUNK_BY_PREFIX
    if (P.outSAMorder != "PairedKeepInputOrder")
        sortChunkByPrefix(chunkIn, chunkInSizeBytesTotal, P.readNends);
    #endif

    #ifdef OPT_DEDUP_R2
    RA->dedupResetCache();
    #endif
    
    RA->statsRA.resetN();

    for (uint ii=0;ii<P.readNends;ii++) {//clear eof and rewind the input streams
        RA->readInStream[ii]->clear();
        RA->readInStream[ii]->seekg(0,ios::beg);
    };
    
    

    if ( P.outSAMorder == "PairedKeepInputOrder" && P.runThreadN>1 ) {//open chunk file
        ostringstream name1("");
        name1 << P.outFileTmp + "/Aligned.tmp.sam.chunk"<<iChunkIn;
        chunkOutBAMfileName = name1.str();
        chunkOutBAMfile.open(chunkOutBAMfileName.c_str());
    };

    int readStatus=0;
    while (readStatus==0) {//main cycle over all reads

        readStatus=RA->oneRead(); //map one read

        if (readStatus==0) {//there was a read processed
            RA->iRead++;
//         chunkOutBAMtotal=(uint) RA->outSAMstream->tellp();
            chunkOutBAMtotal+=RA->outBAMbytes;
//             uint ddd=(uint) RA->outSAMstream->tellp();
        };

        //write SAM aligns to chunk buffer
        if (P.outSAMbool) {
            if ( chunkOutBAMtotal > P.chunkOutBAMsizeBytes ) {//this should not happen!
                ostringstream errOut;
                errOut <<"EXITING because of fatal error: buffer size for SAM/BAM output is too small\n";
                errOut <<"Solution: increase input parameter --limitOutSAMoneReadBytes\n";
                exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
            } else if ( chunkOutBAMtotal + P.limitOutSAMoneReadBytes > P.chunkOutBAMsizeBytes || (readStatus==-1 && noReadsLeft) ) {//write buffer to disk because it's almost full, or all reads are mapped
                if ( P.outSAMorder == "PairedKeepInputOrder" && P.runThreadN>1 ) {//output chunks into separate files
                    chunkOutBAMfile.write(chunkOutBAM,chunkOutBAMtotal);
                    chunkOutBAMfile.clear(); //in case 0 bytes were written which could set fail bit
                    //chunkOutBAMfile.flush(); //not needed
                } else {//standard way, directly into Aligned.out.sam file
                    //SAM output
                    if (P.runThreadN>1) pthread_mutex_lock(&g_threadChunks.mutexOutSAM);
                    P.inOut->outSAM->write(chunkOutBAM,chunkOutBAMtotal);
                    P.inOut->outSAM->clear();//in case 0 bytes were written which could set fail bit
                    //P.inOut->outSAM->flush(); //not needed
                    if (P.runThreadN>1) pthread_mutex_unlock(&g_threadChunks.mutexOutSAM);
                };
                RA->outSAMstream->seekp(0,ios::beg); //rewind the chunk storage
                chunkOutBAMtotal=0;
            };
        };

        //collapse SJ buffer if needed
        if ( !P.outSJ.yes ) {
            //do nothing
        } else if ( chunkOutSJ->N > chunkOutSJ->Nstore ) {//this means the number of collapsed junctions is larger than the chunks size
            ostringstream errOut;
            errOut <<"EXITING because of fatal error: buffer size for SJ output is too small\n";
            errOut <<"Solution: increase input parameter --limitOutSJoneRead\n";
            exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        } else if ( chunkOutSJ->N + P.limitOutSJoneRead > chunkOutSJ->Nstore || (readStatus==-1 && noReadsLeft) ) {//write buffer to disk because it's almost full, or all reads are mapped
            chunkOutSJ->collapseSJ();
            if ( chunkOutSJ->N + 2*P.limitOutSJoneRead > chunkOutSJ->Nstore ) {
                /*
                ostringstream errOut;
                errOut <<"EXITING because of fatal error: buffer size for SJ output is too small\n";
                errOut <<"Solution: increase input parameter --limitOutSJcollapsed\n";
                exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
                */
                chunkOutSJ->dataSizeIncrease();
                P.inOut->logMain << "Increased the size of chunkOutSJ to " << chunkOutSJ->Nstore <<'\n';
            };
        };

        //collapse SJ1 buffer if needed
        if ( P.outFilterBySJoutStage != 1 ) {//no outFilterBySJoutStage
            //do nothing
        } else if ( chunkOutSJ1->N > chunkOutSJ->Nstore ) {//this means the number of collapsed junctions is larger than the chunks size
            ostringstream errOut;
            errOut <<"EXITING because of fatal error: buffer size for SJ output is too small\n";
            errOut <<"Solution: increase input parameter --limitOutSJoneRead\n";
            exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        } else if ( chunkOutSJ1->N + P.limitOutSJoneRead > chunkOutSJ->Nstore || (readStatus==-1 && noReadsLeft) ) {//write buffer to disk because it's almost full, or all reads are mapped
            chunkOutSJ1->collapseSJ();
            if ( chunkOutSJ1->N + 2*P.limitOutSJoneRead > chunkOutSJ->Nstore ) {
                /*
                ostringstream errOut;
                errOut <<"EXITING because of fatal error: buffer size for SJ output is too small\n";
                errOut <<"Solution: increase input parameter --limitOutSJcollapsed\n";
                exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
                */
                chunkOutSJ->dataSizeIncrease();
                P.inOut->logMain << "Increased the size of chunkOutSJ to " << chunkOutSJ->Nstore <<'\n';
            };
        };

    }; //reads cycle

    if ( P.outSAMbool && P.outSAMorder == "PairedKeepInputOrder" && P.runThreadN>1 ) {//write the remaining part of the buffer, close and rename chunk files
        chunkOutBAMfile.write(chunkOutBAM,chunkOutBAMtotal);
        chunkOutBAMfile.clear(); //in case 0 bytes were written which could set fail bit
        chunkOutBAMfile.close();
        RA->outSAMstream->seekp(0,ios::beg); //rewind the chunk storage
        chunkOutBAMtotal=0;
        ostringstream name2("");
        name2 << P.outFileTmp + "/Aligned.out.sam.chunk"<<iChunkIn;
        rename(chunkOutBAMfileName.c_str(),name2.str().c_str());//marks files as completedly written
    };

    //add stats, write progress if needed
    if (P.runThreadN>1) pthread_mutex_lock(&g_threadChunks.mutexStats);
    g_statsAll.addStats(RA->statsRA);
    g_statsAll.progressReport(P.inOut->logProgress);
    if (P.runThreadN>1) pthread_mutex_unlock(&g_threadChunks.mutexStats);
};
