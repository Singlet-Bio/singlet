#include "BAMbinSortByCoordinate.h"
#include "ErrorWarning.h"
#include "serviceFuns.cpp"
#include "BAMfunctions.h"
#include "SequenceFuns.h"
#include <zlib.h>
#include <cstring>

// Load all alignments of one coordinate bin from the per-thread temporary files and sort them by
// (coordinate, read order). Fills bamIn (binS bytes) and startPos (3 x binN: coordinate, iRead, offset).
static void loadAndSortBin(uint32 iBin, uint binN, uint binS, uint nThreads, const string &dirBAMsort, Parameters &P,
                           char *bamIn, uint *startPos) {
    uint bamInBytes=0;
    //load all aligns
    for (uint it=0; it<nThreads; it++) {
        string bamInFile=dirBAMsort+to_string(it)+"/"+to_string((uint) iBin);
        ifstream bamInStream;
        bamInStream.open(bamInFile.c_str(),std::ios::binary | std::ios::ate);//open at the end to get file size
        int64 s1=bamInStream.tellg();
        if (s1>0)         {
            bamInStream.seekg(std::ios::beg);
            bamInStream.read(bamIn+bamInBytes,s1);//read the whole file
        } else if (s1<0) {
            ostringstream errOut;
            errOut << "EXITING because of FATAL ERROR: failed reading from temporary file: " << dirBAMsort+to_string(it)+"/"+to_string((uint) iBin);
            exitWithError(errOut.str(),std::cerr, P.inOut->logMain, 1, P);
        };
        bamInBytes += bamInStream.gcount();
        bamInStream.close();
        remove(bamInFile.c_str());
    };
    if (bamInBytes!=binS) {
        ostringstream errOut;
        errOut << "EXITING because of FATAL ERROR: number of bytes expected from the BAM bin does not agree with the actual size on disk: ";
        errOut << "Expected bin size=" <<binS <<" ; size on disk="<< bamInBytes <<" ; bin number="<< iBin <<"\n";
        exitWithError(errOut.str(),std::cerr, P.inOut->logMain, 1, P);
    };

    //extract coordinates
    for (uint ib=0,ia=0;ia<binN;ia++) {
        uint32 *bamIn32=(uint32*) (bamIn+ib);
        startPos[ia*3]  =( ((uint) bamIn32[1]) << 32) | ( (uint)bamIn32[2] );
        startPos[ia*3+2]=ib;
        ib+=bamIn32[0]+sizeof(uint32);//note that size of the BAM record does not include the size record itself
        startPos[ia*3+1]=*( (uint*) (bamIn+ib) ); //read order
        ib+=sizeof(uint);
    };

    //sort
    qsort((void*) startPos, binN, sizeof(uint)*3, funCompareArrays<uint,3>);
};

void BAMbinSortByCoordinate(uint32 iBin, uint binN, uint binS, uint nThreads, string dirBAMsort, Parameters &P, Genome &genome, Solo &solo) {

    if (binS==0) return; //nothing to do for empty bins
    //allocate arrays
    char *bamIn=new char[binS+1];
    uint *startPos=new uint[binN*3];

    loadAndSortBin(iBin, binN, binS, nThreads, dirBAMsort, P, bamIn, startPos);

    BGZF *bgzfBin;
    bgzfBin=bgzf_open((dirBAMsort+"/b"+to_string((uint) iBin)).c_str(),("w"+to_string((long long) P.outBAMcompression)).c_str());
    if (bgzfBin==NULL) {
        ostringstream errOut;
        errOut <<"EXITING because of fatal ERROR: could not open temporary bam file: " << dirBAMsort+"/b"+to_string((uint) iBin) << "\n";
        errOut <<"SOLUTION: check that the disk is not full, increase the max number of open files with Linux command ulimit -n before running STAR";
        exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
    };

    outBAMwriteHeader(bgzfBin,P.samHeaderSortedCoord,genome.chrNameAll,genome.chrLengthAll);
    //send ordered aligns to bgzf one-by-one
    char bam1[BAM_ATTR_MaxSize];//temp array
    for (uint ia=0;ia<binN;ia++) {
        char* bam0=bamIn+startPos[ia*3+2];
        uint32 size0=*((uint32*) bam0)+sizeof(uint32);

        if (solo.pSolo.samAttrYes)
            solo.soloFeat[solo.pSolo.featureInd[solo.pSolo.samAttrFeature]]->addBAMtags(bam0,size0,bam1);

        bgzf_write(bgzfBin, bam0, size0);
    };

    bgzf_flush(bgzfBin);
    bgzf_close(bgzfBin);
    //release memory
    delete [] bamIn;
    delete [] startPos;
};

// ── In-memory BGZF writer ─────────────────────────────────────────────────────────────────────────────────────────
// Same block layout and compression as htslib's bgzf_write()/bgzf_flush() in the bundled bgzf.c (blocks of
// BGZF_BLOCK_SIZE uncompressed bytes, raw deflate at the given level, crc32 + isize footer), but into a memory
// buffer, so a sorted bin can be compressed by its worker and appended to the final BAM with bgzf_raw_write().
namespace {
const uint8_t kBgzfMagic[18] = {037,0213,010,4,0,0,0,0,0,0377,6,0,0102,0103,2,0,0,0};

struct MemBgzf {
    int level;
    std::vector<char> &out;
    char stage[BGZF_BLOCK_SIZE];
    int pos=0;
    z_stream zs;
    bool zinit=false;
    MemBgzf(int lev, std::vector<char> &o) : level(lev<0 ? Z_DEFAULT_COMPRESSION : lev), out(o) {};
    ~MemBgzf() { if (zinit) deflateEnd(&zs); };
    bool flushBlock() {
        if (pos==0) return true;
        size_t o0=out.size();
        out.resize(o0+BGZF_MAX_BLOCK_SIZE);
        uint8_t *dst=(uint8_t*) out.data()+o0;
        if (!zinit) {
            memset(&zs,0,sizeof(zs));
            if (deflateInit2(&zs, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) return false;
            zinit=true;
        } else if (deflateReset(&zs) != Z_OK) {
            return false;
        };
        zs.next_in=(Bytef*) stage;
        zs.avail_in=pos;
        zs.next_out=dst+18;
        zs.avail_out=BGZF_MAX_BLOCK_SIZE-18-8;
        if (deflate(&zs, Z_FINISH) != Z_STREAM_END) return false;
        int dlen=(int) zs.total_out+18+8;
        memcpy(dst, kBgzfMagic, 18);
        dst[16]=(uint8_t) ((dlen-1) & 0xff); dst[17]=(uint8_t) ((dlen-1) >> 8);
        uint32_t crc=crc32(crc32(0L, NULL, 0L), (Bytef*) stage, pos);
        uint32_t isz=(uint32_t) pos;
        for (int k=0;k<4;k++) { dst[dlen-8+k]=(uint8_t) (crc>>(8*k)); dst[dlen-4+k]=(uint8_t) (isz>>(8*k)); };
        out.resize(o0+dlen);
        pos=0;
        return true;
    };
    bool write(const char *d, size_t n) {
        while (n>0) {
            size_t k=std::min(n, (size_t) (BGZF_BLOCK_SIZE-pos));
            memcpy(stage+pos, d, k);
            pos+=k; d+=k; n-=k;
            if (pos==BGZF_BLOCK_SIZE && !flushBlock()) return false;
        };
        return true;
    };
};
};

void BAMbinSortByCoordinateToBuffer(uint32 iBin, uint binN, uint binS, uint nThreads, string dirBAMsort, Parameters &P,
                                    Genome &genome, Solo &solo, std::vector<char> &out) {
    out.clear();
    if (binS==0) return;
    char *bamIn=new char[binS+1];
    uint *startPos=new uint[binN*3];

    loadAndSortBin(iBin, binN, binS, nThreads, dirBAMsort, P, bamIn, startPos);

    // level 0 stores: output ~ input; otherwise reserve a third and let it grow
    out.reserve(P.outBAMcompression==0 ? binS+binS/64+BGZF_MAX_BLOCK_SIZE : binS/3+BGZF_MAX_BLOCK_SIZE);
    {
        MemBgzf bg(P.outBAMcompression, out);
        char bam1[BAM_ATTR_MaxSize];//temp array
        bool ok=true;
        for (uint ia=0;ia<binN && ok;ia++) {
            char* bam0=bamIn+startPos[ia*3+2];
            uint32 size0=*((uint32*) bam0)+sizeof(uint32);
            if (solo.pSolo.samAttrYes)
                solo.soloFeat[solo.pSolo.featureInd[solo.pSolo.samAttrFeature]]->addBAMtags(bam0,size0,bam1);
            ok=bg.write(bam0, size0);
        };
        if (ok) ok=bg.flushBlock();
        if (!ok) {
            ostringstream errOut;
            errOut <<"EXITING because of fatal ERROR: BGZF compression failed for BAM bin " << iBin << "\n";
            exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_BUG, P);
        };
    };
    delete [] bamIn;
    delete [] startPos;
};
