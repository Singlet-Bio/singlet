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
    f(t.maxScore); f(t.iFrag); f(t.haploType); f(t.sjMotifStrand); f(t.sjYes); f(t.primaryFlag);
}
template <class F> void forEachRow(Transcript &t, uint64 iex, F f) {
    for (int k=0; k<EX_SIZE; k++) f(t.exons[iex][k]);
    f(t.shiftSJ[iex][0]); f(t.shiftSJ[iex][1]);
    f(t.canonSJ[iex]); f(t.sjAnnot[iex]); f(t.sjStr[iex]);
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
