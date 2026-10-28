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
#include <cstring>

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
    const uint8 *p = blob.data();
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
void ReadAlign::seqCacheAlignRecords(const char *seqNum, uint L, const std::vector<uint8> &blob, std::vector<uint8> &out) {
    Lread=L; readLength[0]=L; readLength[1]=0; readLengthOriginal[0]=L; readLengthOriginal[1]=0; readLengthPairOriginal=L;
    memcpy(Read1[0], seqNum, L);
    complementSeqNumbers(Read1[0],Read1[1],Lread);
    for (uint ii=0;ii<Lread;ii++) Read1[2][Lread-ii-1]=Read1[1][ii];
    static const char NUC[5]={'A','C','G','T','N'};
    for (uint ii=0;ii<L;ii++) {Read0[0][ii]=NUC[(uint8)seqNum[ii]<4 ? (uint8)seqNum[ii] : 4]; Qual0[0][ii]='I';};
    Read0[0][L]=0; Qual0[0][L]=0;
    strcpy(readName,"@x"); readFilter='N'; readNameExtra[0].clear();
    seqCacheRestore(blob);
    mappedFilter();
    const size_t at=out.size(); out.push_back(0);
    uint8 nAln=0;
    if (unmapType<0) {
        static const vector<int> attrs={ATTR_NH, ATTR_nM};
        const uint64 nW=min((uint64) P.outSAMmultNmax, (uint64) nTr);
        for (uint64 iTr=0; iTr<nW; iTr++) {
            alignBAM(*trMult[iTr], nTr, iTr, mapGen.chrStart[trMult[iTr]->Chr], (uint) -1, (uint) -1, 0, -1, NULL, attrs, outBAMoneAlign, outBAMoneAlignNbytes);
            const uint8 *b=(const uint8*) outBAMoneAlign[0];
            uint32 w[9]; memcpy(w, b, 36);
            const uint32 lname=w[3]&0xff, ncig=w[4]&0xffff, lseq=w[5];
            const uint8 *cig=b+36+lname, *aux=cig+4*ncig+(lseq+1)/2+lseq, *end=b+4+w[0];
            uint32 nh=0, nm=0;
            for (const uint8 *a=aux; a+3<=end; ) {//NH, nM: integer tags (STAR writes them as C / S / I)
                const char t0=(char) a[0], t1=(char) a[1], ty=(char) a[2]; const uint8 *v=a+3; uint32 val=0; int width=0;
                if (ty=='C' || ty=='c') {val=v[0]; width=1;} else if (ty=='S' || ty=='s') {uint16 x; memcpy(&x,v,2); val=x; width=2;}
                else if (ty=='I' || ty=='i') {uint32 x; memcpy(&x,v,4); val=x; width=4;} else break;
                if (t0=='N' && t1=='H') nh=val; else if (t0=='n' && t1=='M') nm=val;
                a=v+width;
            };
            auto put=[&](const void *x, size_t n) {out.insert(out.end(), (const uint8*) x, (const uint8*) x+n);};
            int32 refID=(int32) w[1], pos=(int32) w[2]; uint16 flag=(uint16) (w[4]>>16); uint8 mapq=(uint8) ((w[3]>>8)&0xff), z=0;
            uint16 nh16=(uint16) min(nh,(uint32) 65535), nm16=(uint16) min(nm,(uint32) 65535), nc16=(uint16) ncig;
            put(&refID,4); put(&pos,4); put(&flag,2); put(&mapq,1); put(&z,1); put(&nh16,2); put(&nm16,2); put(&nc16,2); put(cig,4*ncig);
            nAln++;
        };
    };
    out[at]=nAln;
    static const uint8 NT16[5]={1,2,4,8,15};
    uint16 l16=(uint16) L; out.insert(out.end(), (const uint8*) &l16, (const uint8*) &l16+2);
    for (uint ii=0; ii<L; ii+=2) {
        uint8 lo=NT16[(uint8)seqNum[ii]<4 ? (uint8)seqNum[ii] : 4], hi=ii+1<L ? NT16[(uint8)seqNum[ii+1]<4 ? (uint8)seqNum[ii+1] : 4] : 0;
        out.push_back((uint8) (lo | (hi<<4)));
    };
}
