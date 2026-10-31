#include "SoloFeature.h"
#include "streamFuns.h"
#include "TimeFunctions.h"
#include "SequenceFuns.h"
#include "systemFunctions.h"
#include <thread>
#include "PhaseMark.h"

void SoloFeature::countCBgeneUMI()
{    
    time_t rawTime;
    
    rguStride=2;
    if (pSolo.readIndexYes[featureType])
        rguStride=3; //to keep readI column

    if (pSolo.readInfoYes[featureType]) {
        readInfo.resize(nReadsInput,{(uint64)-1,(uint32)-1});
        time(&rawTime);
        P.inOut->logMain << timeMonthDayTime(rawTime) << " ... Allocated and initialized readInfo array, nReadsInput = " << nReadsInput <<endl;        
    };

    rGeneUMI = new uint32[rguStride*nReadsMapped]; //big array for all CBs - each element is gene and UMI
    rCBp = new uint32*[nCB+1];
    uint32 **rCBpa = new uint32*[pSolo.cbWLsize+1];
    
    rCBp[0]=rGeneUMI;
    rCBpa[0]=rGeneUMI;
    nCB=0;//will count it again below
    for (uint32 ii=0; ii<pSolo.cbWLsize; ii++) {
        if (readFeatSum->cbReadCount[ii]>0) {
            rCBp[nCB+1] = rCBp[nCB] + rguStride*readFeatSum->cbReadCount[ii];
            ++nCB;
        };
        rCBpa[ii+1]=rCBp[nCB];
    };

    //read and store the CB/gene/UMI from files
    time(&rawTime);
    P.inOut->logMain << timeMonthDayTime(rawTime) << " ... Finished allocating arrays for Solo " << nReadsMapped*rguStride*4.0/1024/1024/1024 <<" GiB" <<endl;

    ///////////////////////////////////////////////////////////////////////////
    ////////////// Input records
    readFlagCounts.flagCounts.reserve(nCB*3/2);
    readFlagCounts.flagCountsNoCB = {};
    vector<uint32> nReadPerCBunique1(pSolo.cbWLsize), nReadPerCBmulti1(pSolo.cbWLsize); //temp arrays to record # of reads for all cells in the WL
    //singlet --engine gpu (SINGLET_SOLO_PAR=0 turns it off): the threads' record files are read in parallel. Each thread keeps its counted records in
    //file order; they are then placed at the CB cursors thread by thread, which is the layout the serial loop makes, and the flag counts are
    //merged with their keys inserted in the order the serial loop first inserts them, so every later step sees the same data.
    const char *soloPar = getenv("SINGLET_SOLO_PAR");
    if (getenv("SINGLET_ENGINE_GPU") != nullptr && !(soloPar != nullptr && soloPar[0] == '0') && P.runThreadN > 1) {
        const int T = P.runThreadN;
        const uint64 W = pSolo.cbWLsize;
        vector<SoloRecSink> sinks(T);
        vector<SoloReadFlagClass> flags(T);
        vector<vector<uintCB>> orders(T);
        vector<vector<uint32>> nU(T), nM(T);
        {
            vector<std::thread> th;
            for (int t=0; t<T; t++)
                th.emplace_back([&, t]() {
                    sinks[t].stride = rguStride; sinks[t].cnt.assign(W, 0);
                    flags[t].order = &orders[t];
                    nU[t].assign(W, 0); nM[t].assign(W, 0);
                    readFeatAll[t]->inputRecords(rCBpa, rguStride, readBarSum->cbReadCountExact, readInfo, flags[t], nU[t], nM[t], &sinks[t]);
                });
            for (auto &x : th) x.join();
        };
        //where each CB's region starts; then, per CB, each thread's first slot (in cnt) and the cursor at the end of the region
        vector<uint32*> starts(rCBpa, rCBpa + W);
        {
            vector<std::thread> th;
            for (int t=0; t<T; t++)
                th.emplace_back([&, t]() {
                    for (uint64 cb = W*t/T; cb < W*(t+1)/T; cb++) {
                        uint32 run = 0;
                        for (int k=0; k<T; k++) {const uint32 c = sinks[k].cnt[cb]; sinks[k].cnt[cb] = run; run += c;};
                        rCBpa[cb] = starts[cb] + (uint64) rguStride * run;
                    };
                });
            for (auto &x : th) x.join();
        };
        {
            vector<std::thread> th;
            for (int t=0; t<T; t++)
                th.emplace_back([&, t]() {
                    const vector<uint32> &v = sinks[t].v;
                    const uint64 rs = rguStride + 1;
                    for (uint64 i = 0; i < v.size(); i += rs) {
                        uint32 *d = starts[v[i]] + (uint64) rguStride * (sinks[t].cnt[v[i]]++);
                        for (uint32 j = 0; j < rguStride; j++)
                            d[j] = v[i + 1 + j];
                    };
                    vector<uint32>().swap(sinks[t].v);
                });
            for (auto &x : th) x.join();
        };
        for (int t=0; t<T; t++) {
            for (uintCB cb : orders[t]) {
                auto ins = readFlagCounts.flagCounts.insert({cb, {}});
                const auto &src = flags[t].flagCounts[cb];
                for (uint32 ib=0; ib<readFlagCounts.nBits; ib++)
                    ins.first->second[ib] += src[ib];
            };
            readFlagCounts.countsAddNoCBarray(flags[t].flagCountsNoCB);
            for (uint64 cb=0; cb<W; cb++) {nReadPerCBunique1[cb] += nU[t][cb]; nReadPerCBmulti1[cb] += nM[t][cb];};
            readFeatSum->addStats(*readFeatAll[t]);
        };
    } else
    for (int ii=0; ii<P.runThreadN; ii++) {//TODO: this can be parallelized
        readFeatAll[ii]->inputRecords(rCBpa, rguStride, readBarSum->cbReadCountExact, readInfo, readFlagCounts, nReadPerCBunique1, nReadPerCBmulti1);
        readFeatSum->addStats(*readFeatAll[ii]);//sum stats: has to be done after inputRecords, since the stats values are updated there
    };
    readFlagCounts.countsAddNoCBarray(readFeatSum->readFlag.flagCountsNoCB);//add no-CB counts calculated in SoloReadFeature_record.cpp and not recorded to temp Solo files

    starPhaseMark(P, "solo_records_read");
    nReadPerCBtotal.resize(nCB);
    nReadPerCBunique.resize(nCB);
    for (uint32 icb=0; icb<nCB; icb++) {
        nReadPerCBunique[icb] = nReadPerCBunique1[indCB[icb]];
        nReadPerCBtotal[icb] = nReadPerCBunique[icb] + nReadPerCBmulti1[indCB[icb]];
    };

    //debug
    /*{
        uint64 n1=0,n2=0;
        for (uint32 icb=0; icb<nCB; icb++) {
            n1 += nReadPerCBtotal[icb];
        };

        for (auto &c: readFlagCounts.flagCounts) {
            n2+= c.second[readFlagCounts.countedU] + c.second[readFlagCounts.countedM];
        };

        cout << "n1,2=" << n1<<" "<<n2<<endl;
    };*/

    nReadPerCB.resize(nCB);
    nReadPerCBmax=0;
    for (uint32 iCB=0; iCB<nCB; iCB++) {
        nReadPerCB[iCB] = (rCBpa[indCB[iCB]]-rCBp[iCB])/rguStride;  //number of reads that were matched to WL, rCBpa accumulated reference to the last element+1
                                                                    //for multimappers this is the number of all alignments > number of reads
        nReadPerCBmax=max(nReadPerCBmax,nReadPerCB[iCB]);
        //readFeatSum->stats.V[readFeatSum->stats.yesWLmatch] += nReadPerCB[iCB];
    };    
    
    time(&rawTime);
    P.inOut->logMain << timeMonthDayTime(rawTime) << " ... Finished reading reads from Solo files nCB="<<nCB <<", nReadPerCBmax="<<nReadPerCBmax;
    P.inOut->logMain <<", yesWLmatch="<<readFeatSum->stats.V[readFeatSum->stats.yesWLmatch]<<endl;
    
    //////////////////////////////////////////////////////////////////////////////
    /////////////////////////// collapse each CB
    nUMIperCB.resize(nCB);
    nGenePerCB.resize(nCB);
    
                     //dedup options        //gene ID
    countMatStride = pSolo.umiDedup.yes.N + 1;
    countCellGeneUMI.resize(nReadsMapped*countMatStride/5+16); //5 is heuristic, will be resized if needed
    countCellGeneUMIindex.resize(nCB+1, 0);
    
    if (pSolo.multiMap.yes.multi) {
                    //gene   //uniform  //rescue
        countMatMult.s = 1 + pSolo.multiMap.yes.N * pSolo.umiDedup.yes.N;
        countMatMult.m.resize(nReadsMapped*countMatMult.s/5+16);
        countMatMult.i.resize(nCB+1, 0);
    };

    collapseUMIall();
    starPhaseMark(P, "solo_umi_collapsed");
        
    P.inOut->logMain << "RAM for solo feature "<< SoloFeatureTypes::Names[featureType] <<"\n"
                     <<  linuxProcMemory() << flush;        
    delete[] rGeneUMI;
    delete[] rCBp;
    delete[] rCBpa;
    
    time(&rawTime);
    P.inOut->logMain << timeMonthDayTime(rawTime) << " ... Finished collapsing UMIs" <<endl;
};
