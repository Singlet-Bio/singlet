// Encoding of a read's mapping result for the sequence cache (SeqCache.h).
//
// After mapOneRead + multMapSelect + mappedFilter + transformGenome + chimericDetection, the state
// that the rest of oneRead() (output, Solo, stats) reads is: nW, nTr, trMult[0..nTr), trBest (one
// of trMult, or trInit when nW==0), chimRecord (false: no chimeric detection) and peOv.yes (false:
// single-end). Every transcript recorded by stitching starts as a copy of trInit, which mapOneRead
// initialises from the current read; restoring re-does that initialisation, copies trInit into the
// transcript slot and then overwrites every field that the mapping computed. Fields are stored as
// zigzag LEB128 varints (most are small; coordinates take ~5 bytes).

#include "ReadAlign.h"
#include "SeqCache.h"
#include "SequenceFuns.h"
#include "Transcriptome.h"
#include "OutSJ.h"
#include "Stats.h"
#include <cstring>
#include <cstdlib>
#include <atomic>

namespace {
inline void putV(std::vector<uint8> &b, int64 v) {
    uint64 u = ((uint64) v << 1) ^ (uint64) (v >> 63);
    while (u >= 0x80) {b.push_back((uint8) (u | 0x80)); u >>= 7;}
    b.push_back((uint8) u);
}
inline int64 getV(const uint8 *&p) {
    uint64 u = 0; int sh = 0;
    while (*p & 0x80) {u |= (uint64) (*p++ & 0x7f) << sh; sh += 7;}
    u |= (uint64) (*p++) << sh;
    return (int64) ((u >> 1) ^ (~(u & 1) + 1));
}

//every Transcript field written by seed stitching / multMapSelect; order matters (encode == decode)
template <class F> void forEachScalar(Transcript &t, F f) {
    f(t.rStart); f(t.roStart); f(t.rLength); f(t.gStart); f(t.gLength); f(t.cStart);
    f(t.Chr); f(t.Str); f(t.roStr);
    f(t.nMatch); f(t.nMM); f(t.mappedLength); f(t.extendL);
    f(t.nGap); f(t.lGap); f(t.nDel); f(t.nIns); f(t.lDel); f(t.lIns); f(t.nUnique); f(t.nAnchor);
    f(t.intronMotifs[0]); f(t.intronMotifs[1]); f(t.intronMotifs[2]);
    f(t.maxScore); f(t.iFrag); f(t.sjMotifStrand); f(t.sjYes); f(t.primaryFlag);
    //not stored: haploType (set only by genome transformation, off here: uninitialised otherwise)
}
template <class F> void forEachRow(Transcript &t, uint64 iex, F f) {
    for (int k=0; k<EX_SIZE; k++) f(t.exons[iex][k]);
    if (iex+1 < t.nExons) {//junction iex joins exons iex and iex+1; the last exon has none (its slots are left uninitialised by stitching)
        f(t.canonSJ[iex]); f(t.sjAnnot[iex]); f(t.sjStr[iex]);
        if (t.canonSJ[iex] >= 0) {//shiftSJ is set for splice junctions only, not for indels (canonSJ<0)
            f(t.shiftSJ[iex][0]); f(t.shiftSJ[iex][1]);
        };
    };
}
struct Put {
    std::vector<uint8> &b;
    template <class T> void operator()(T &x) {putV(b, (int64) x);}
    void operator()(bool &x) {putV(b, x ? 1 : 0);}
};
struct Get {
    const uint8 *&p;
    template <class T> void operator()(T &x) {x = (T) getV(p);}
    void operator()(bool &x) {x = getV(p) != 0;}
};
}

bool ReadAlign::seqCacheable() {
    if (Lread == 0 || Lread > DEF_readSeqLengthMax)
        return false;
    for (uint64 i=0; i<Lread; i++)
        if ((uint8) Read1[0][i] > 3) //N: the key is 2-bit packed
            return false;
    return true;
}

bool ReadAlign::seqCacheEncode(std::vector<uint8> &b) {
    b.clear();
    if (chimRecord || peOv.yes || nTr > 255)
        return false;
    int64 iBest = -1;
    if (nW == 0) {
        if (trBest != trInit) return false;
    } else {
        for (uint64 i=0; i<nTr; i++)
            if (trMult[i] == trBest) {iBest = (int64) i; break;}
        if (iBest < 0) return false;
    };
    putV(b, (int64) nW);
    putV(b, (int64) nTr);
    putV(b, iBest);
    if (nW == 0)
        putV(b, (int64) trInit->rLength);
    Put put{b};
    for (uint64 i=0; i<nTr; i++) {
        Transcript &t = *trMult[i];
        if (!t.cigar.empty() || !t.varInd.empty() || !t.varGenCoord.empty() || !t.varReadCoord.empty()
            || !t.varAllele.empty() || !t.alignGenes.empty() || t.nExons > MAX_N_EXONS)
            return false;
        putV(b, (int64) t.nExons);
        forEachScalar(t, put);
        for (uint64 iex=0; iex<t.nExons; iex++)
            forEachRow(t, iex, put);
    };
    return true;
}

void ReadAlign::seqCacheRestore(const std::vector<uint8> &blob) {
    seqCacheRestore(blob.data());
}

void ReadAlign::seqCacheRestore(const uint8 *p) {
    //as in mapOneRead()
    resetN();
    trInit->reset();
    trInit->Chr=0;    trInit->Str=0; trInit->roStr=0;    trInit->cStart=0;     trInit->gLength=0;
    trInit->iRead=iRead;
    trInit->Lread=Lread;
    trInit->nExons=0;
    trInit->readLengthOriginal=readLengthOriginal;
    trInit->readLengthPairOriginal=readLengthPairOriginal;
    trInit->readLength=readLength;
    trInit->readNmates=readNmates;
    trInit->readName=readName;

    nW = (uint) getV(p);
    nTr = (uint64) getV(p);
    int64 iBest = getV(p);
    if (nW == 0)
        trInit->rLength = (uint) getV(p);
    Get get{p};
    for (uint64 i=0; i<nTr; i++) {
        Transcript &t = trArray[i];
        t = *trInit;
        t.nExons = (uint) getV(p);
        forEachScalar(t, get);
        for (uint64 iex=0; iex<t.nExons; iex++)
            forEachRow(t, iex, get);
        trMult[i] = &t;
    };
    trBest = iBest >= 0 ? trMult[iBest] : trInit;
    if (nW > 0)
        maxScore = trBest->maxScore;
    chimRecord = false;
    peOv.yes = false;
}

//singlet --engine gpu (GPU pileup hand-off): what STAR's own per-read output computes for a read with this sequence, from the cached
//mapping state, as oneRead + writeSAM + alignBAM do for a cache hit. Per record: uint8 n alignments, then per alignment int32 refID,
//int32 pos, uint16 flag, uint8 MAPQ, uint8 0, uint16 NH, uint16 nM, uint16 n_cigar, n_cigar uint32 BAM CIGAR operations (all read from
//the BAM bytes alignBAM writes, so they are STAR's by construction), then uint16 read length and the sequence as BAM nt16 codes, two per
//byte, low nibble first, in read orientation.
//The BAM core fields of an alignment, computed from the transcript the way alignBAM does for a single-end mapped read (alignType -1, one
//mate; the same expressions, in the same types), instead of building the whole BAM record and reading the fields back. alignBAM also
//reverse-complements and packs the read and its qualities, writes the attribute array and allocates five vectors, for fields this
//function does not need. Returns false for a transcript with a mate gap (canonSJ -3, paired reads only): the caller then uses alignBAM.
//Checked against a transcription of alignBAM on the laptop by bench/host_cost/model_lean_aln.py; SINGLET_HANDOFF_LEAN=verify runs both
//for every alignment of a run and counts the differences (g_handoffLeanBad, logged by the prefill), =0 uses alignBAM throughout.
static inline int handoffLeanMode() {
    static const int m = [] {
        const char *e = getenv("SINGLET_HANDOFF_LEAN");
        return e == nullptr ? 1 : (strcmp(e, "0") == 0 ? 0 : (strcmp(e, "verify") == 0 ? 2 : 1));
    }();
    return m;
}
std::atomic<unsigned long long> g_handoffLeanAln(0), g_handoffLeanBad(0);

bool ReadAlign::seqCacheLeanAln(Transcript const &trOut, uint nTrOut, uint trChrStart, int32 &refID, int32 &pos, uint16 &flag, uint8 &mapq,
                                uint32 &nCig, uint32 *cig) {
    for (uint ii=0; ii+1<trOut.nExons; ii++)
        if (trOut.canonSJ[ii]==-3) return false;
    const uint iEx1=0, iEx2=trOut.nExons-1, leftMate=0;
    uint16 samFLAG=0;
    if (readFilter=='Y') samFLAG|=0x200;
    if (!trOut.primaryFlag) samFLAG|=0x100;
    const uint Mate=trOut.exons[iEx1][EX_iFrag];
    const uint Str=trOut.Str;
    if (Mate==0) {
        samFLAG|= Str*0x10;
    } else {
        samFLAG|= (1-Str)*0x10;
    };
    uint trimL;
    if (Str==0 && Mate==0) {
        trimL=clipMates[Mate][0].clippedN;
    } else if (Str==0 && Mate==1) {
        trimL=clipMates[Mate][1].clippedN;
    } else if (Str==1 && Mate==0) {
        trimL=clipMates[Mate][1].clippedN;
    } else {
        trimL=clipMates[Mate][0].clippedN;
    };
    uint32 n=0;
    const uint trimL1 = trimL + trOut.exons[iEx1][EX_R] - (trOut.exons[iEx1][EX_R]<readLength[leftMate] ? 0 : readLength[leftMate]+1);
    if (trimL1>0)
        cig[n++]=trimL1<<BAM_CIGAR_OperationShift | BAM_CIGAR_S;
    for (uint ii=iEx1; ii<=iEx2; ii++) {
        if (ii>iEx1) {
            uint gapG=trOut.exons[ii][EX_G]-(trOut.exons[ii-1][EX_G]+trOut.exons[ii-1][EX_L]);
            uint gapR=trOut.exons[ii][EX_R]-trOut.exons[ii-1][EX_R]-trOut.exons[ii-1][EX_L];
            if (gapR>0)
                cig[n++]=gapR<<BAM_CIGAR_OperationShift | BAM_CIGAR_I;
            if (trOut.canonSJ[ii-1]>=0 || trOut.sjAnnot[ii-1]==1) {
                cig[n++]=gapG<<BAM_CIGAR_OperationShift | BAM_CIGAR_N;
            } else if (gapG>0) {
                cig[n++]=gapG<<BAM_CIGAR_OperationShift | BAM_CIGAR_D;
            };
        };
        if (trOut.exons[ii][EX_L]>0)
            cig[n++]=trOut.exons[ii][EX_L]<<BAM_CIGAR_OperationShift | BAM_CIGAR_M;
    };
    const uint trimR1=(trOut.exons[iEx1][EX_R]<readLength[leftMate] ? readLengthOriginal[leftMate] : readLength[leftMate]+1+readLengthOriginal[Mate])
        - trOut.exons[iEx2][EX_R]-trOut.exons[iEx2][EX_L] - trimL;
    if (trimR1>0)
        cig[n++]=trimR1<<BAM_CIGAR_OperationShift | BAM_CIGAR_S;
    int MAPQ=P.outSAMmapqUnique;
    if (nTrOut>=5) {
        MAPQ=0;
    } else if (nTrOut>=3) {
        MAPQ=1;
    } else if (nTrOut==2) {
        MAPQ=3;
    };
    refID=(int32) (uint32) trOut.Chr;
    pos=(int32) (uint32) (trOut.exons[iEx1][EX_G] - trChrStart);
    flag=(uint16) ((samFLAG & P.outSAMflagAND) | P.outSAMflagOR);
    mapq=(uint8) (MAPQ & 0xff);
    nCig=n;
    return true;
}

void ReadAlign::seqCacheAlignRecords(const char *seqNum, uint L, const std::vector<uint8> &blob, std::vector<uint8> &out) {
    seqCacheAlignRecords(seqNum, L, blob.data(), out);
}

void ReadAlign::seqCacheAlignRecords(const char *seqNum, uint L, const uint8 *blob, std::vector<uint8> &out) {
    const int lean=handoffLeanMode();
    Lread=L; readLength[0]=L; readLength[1]=0; readLengthOriginal[0]=L; readLengthOriginal[1]=0; readLengthPairOriginal=L;
    memcpy(Read1[0], seqNum, L);
    complementSeqNumbers(Read1[0],Read1[1],Lread);
    for (uint ii=0;ii<Lread;ii++) Read1[2][Lread-ii-1]=Read1[1][ii];
    static const char NUC[5]={'A','C','G','T','N'};
    if (lean!=1) {//only alignBAM reads the sequence and the qualities
        for (uint ii=0;ii<L;ii++) {Read0[0][ii]=NUC[(uint8)seqNum[ii]<4 ? (uint8)seqNum[ii] : 4]; Qual0[0][ii]='I';};
        Read0[0][L]=0; Qual0[0][L]=0;
    };
    strcpy(readName,"@x"); readFilter='N'; readNameExtra[0].clear();
    //as oneRead sets it for every read (mappedFilter's mismatch limit; it was left uninitialised here)
    outFilterMismatchNmaxTotal=min(P.outFilterMismatchNmax, (uint) (P.outFilterMismatchNoverReadLmax*(readLength[0]+readLength[1])));
    seqCacheRestore(blob);
    mappedFilter();
    //the record is built in a thread's buffer and appended to out once (a push_back or insert per field was ~15 calls per alignment)
    static thread_local std::vector<uint8> rec;
    rec.clear();
    rec.push_back(0);
    uint8 nAln=0;
    if (unmapType<0) {
        static const vector<int> attrs={ATTR_NH, ATTR_nM};
        const uint64 nW=min((uint64) P.outSAMmultNmax, (uint64) nTr);
        for (uint64 iTr=0; iTr<nW; iTr++) {
            int32 refID=0, pos=0; uint16 flag=0; uint8 mapq=0; uint32 ncig=0, cigL[BAM_CIGAR_MaxSize]; uint32 nh=(uint32) nTr, nm=(uint32) trMult[iTr]->nMM;
            const bool leanOk = lean!=0 && seqCacheLeanAln(*trMult[iTr], nTr, mapGen.chrStart[trMult[iTr]->Chr], refID, pos, flag, mapq, ncig, cigL);
            if (!leanOk || lean==2) {
                alignBAM(*trMult[iTr], nTr, iTr, mapGen.chrStart[trMult[iTr]->Chr], (uint) -1, (uint) -1, 0, -1, NULL, attrs, outBAMoneAlign, outBAMoneAlignNbytes);
                const uint8 *b=(const uint8*) outBAMoneAlign[0];
                uint32 w[9]; memcpy(w, b, 36);
                const uint32 lname=w[3]&0xff, ncigB=w[4]&0xffff, lseq=w[5];
                const uint8 *cig=b+36+lname, *aux=cig+4*ncigB+(lseq+1)/2+lseq, *end=b+4+w[0];
                uint32 nhB=0, nmB=0;
                for (const uint8 *a=aux; a+3<=end; ) {//NH, nM: integer tags (STAR writes them as C / S / I)
                    const char t0=(char) a[0], t1=(char) a[1], ty=(char) a[2]; const uint8 *v=a+3; uint32 val=0; int width=0;
                    if (ty=='C' || ty=='c') {val=v[0]; width=1;} else if (ty=='S' || ty=='s') {uint16 x; memcpy(&x,v,2); val=x; width=2;}
                    else if (ty=='I' || ty=='i') {uint32 x; memcpy(&x,v,4); val=x; width=4;} else break;
                    if (t0=='N' && t1=='H') nhB=val; else if (t0=='n' && t1=='M') nmB=val;
                    a=v+width;
                };
                if (leanOk) {//verify: the lean fields against alignBAM's
                    g_handoffLeanAln++;
                    if (refID!=(int32) w[1] || pos!=(int32) w[2] || flag!=(uint16) (w[4]>>16) || mapq!=(uint8) ((w[3]>>8)&0xff) || ncig!=ncigB
                        || memcmp(cigL, cig, 4*(size_t) ncigB)!=0 || nh!=nhB || nm!=nmB)
                        g_handoffLeanBad++;
                };
                refID=(int32) w[1]; pos=(int32) w[2]; flag=(uint16) (w[4]>>16); mapq=(uint8) ((w[3]>>8)&0xff); ncig=ncigB;
                memcpy(cigL, cig, 4*(size_t) ncigB); nh=nhB; nm=nmB;
            };
            const uint8 z=0;
            const uint16 nh16=(uint16) min(nh,(uint32) 65535), nm16=(uint16) min(nm,(uint32) 65535), nc16=(uint16) ncig;
            const size_t at=rec.size();
            rec.resize(at+18+4*(size_t) ncig);
            uint8 *d=rec.data()+at;
            memcpy(d,&refID,4); memcpy(d+4,&pos,4); memcpy(d+8,&flag,2); d[10]=mapq; d[11]=z; memcpy(d+12,&nh16,2); memcpy(d+14,&nm16,2); memcpy(d+16,&nc16,2);
            memcpy(d+18,cigL,4*(size_t) ncig);
            nAln++;
        };
    };
    rec[0]=nAln;
    static const uint8 NT16[5]={1,2,4,8,15};
    const size_t at=rec.size();
    rec.resize(at+2+(L+1)/2);
    uint8 *d=rec.data()+at;
    const uint16 l16=(uint16) L; memcpy(d,&l16,2); d+=2;
    for (uint ii=0; ii<L; ii+=2) {
        uint8 lo=NT16[(uint8)seqNum[ii]<4 ? (uint8)seqNum[ii] : 4], hi=ii+1<L ? NT16[(uint8)seqNum[ii+1]<4 ? (uint8)seqNum[ii+1] : 4] : 0;
        *d++=(uint8) (lo | (hi<<4));
    };
    out.insert(out.end(), rec.begin(), rec.end());
}

//singlet GPU pileup "on": the hand-off summary is exact where STAR's per-read output reads only mapping stats, SJ.out.tab and the Solo
//Gene record from a mapped read's alignments (CB_UMI_Simple, the Gene feature alone, single-end, no transformation, chimeric, WASP,
//BySJout, transcriptomic or gene-count output)
bool ReadAlign::seqCacheHandoffConfig(const Parameters &P) {
    return P.pSolo.type==P.pSolo.SoloTypes::CB_UMI_Simple && P.pSolo.nFeatures==1 && P.pSolo.features[0]==SoloFeatureTypes::Gene
        && P.readNmates==1 && P.outFilterBySJoutStage==0 && !P.pGe.transform.outYes && !P.pGe.transform.outSAM && !P.pGe.transform.outQuant
        && P.pCh.segmentMin==0 && !P.wasp.yes && !P.quant.trSAM.yes && !P.quant.geCount.yes && !P.quant.geneFull.yes
        && !P.quant.geneFull_Ex50pAS.yes && !P.quant.geneFull_ExonOverIntron.yes && P.pGe.gType!=101;
}

//Layout: uint16 nTr, uint8 flags (1: an alignment on a mito contig, 2: unique with exons), uint8 Solo Gene ovType, uint16 n genes,
//uint32 genes (fSet order); for nTr==1 Stats::transcriptStats of trMult[0]: uint16 nMM, nIns, nDel, uint32 lIns, lDel, mapped length,
//uint8 splicesN[SJ_MOTIF_SIZE], uint8 splicesNsjdb; then uint16 n and n Junction records as recordSJ writes them (chunkOutSJ is a
//scratch OutSJ here). Called after seqCacheAlignRecords (restored state, mappedFilter done).
void ReadAlign::seqCacheHandoffSummary(std::vector<uint8> &out) {
    if (unmapType>=0 || nTr==0)
        return;
    auto put=[&](const void *x, size_t n) {out.insert(out.end(), (const uint8*) x, (const uint8*) x+n);};
    uint16 n16=(uint16) nTr;
    uint8 flags=0;
    for (uint64 i=0; i<nTr; i++)
        if (P.pGe.chrSet.mito.count(trMult[i]->Chr)==1) flags|=1;
    readAnnot.reset();
    if (P.quant.gene.yes)
        chunkTr->classifyAlign(trMult, nTr, readAnnot);
    const ReadAnnotFeature &af=readAnnot.annotFeatures[SoloFeatureTypes::Gene];
    Stats s;
    s.resetN();
    if (nTr==1) {
        s.transcriptStats(*trMult[0], Lread);
        if (trMult[0]->nExons>0) flags|=2;
    };
    uint8 ov=(uint8) af.ovType;
    uint16 ng=(uint16) af.fSet.size();
    put(&n16,2); put(&flags,1); put(&ov,1); put(&ng,2);
    for (uint32 g : af.fSet) put(&g,4);
    if (nTr==1) {
        uint16 a[3]={(uint16) s.mappedMismatchesN, (uint16) s.mappedInsN, (uint16) s.mappedDelN};
        uint32 b[3]={(uint32) s.mappedInsL, (uint32) s.mappedDelL, (uint32) s.mappedBases};
        uint8 c[SJ_MOTIF_SIZE+1];
        for (int k=0; k<SJ_MOTIF_SIZE; k++) c[k]=(uint8) s.splicesN[k];
        c[SJ_MOTIF_SIZE]=(uint8) s.splicesNsjdb;
        put(a,6); put(b,12); put(c,SJ_MOTIF_SIZE+1);
    };
    chunkOutSJ->N=0;
    recordSJ(nTr, trMult, chunkOutSJ);
    uint16 nsj=(uint16) chunkOutSJ->N;
    put(&nsj,2);
    put(chunkOutSJ->data, (size_t) nsj*Junction::dataSize);
}
