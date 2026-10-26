#include "bamSortByCoordinate.h"
#include "BAMfunctions.h"
#include "BAMbinSortByCoordinate.h"
#include "BAMbinSortUnmapped.h"
#include "ErrorWarning.h"
#include "bam_cat.h"
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdlib>

// The binary links the modern htslib, where bam_hdr_destroy() is an alias of sam_hdr_destroy() (bam_cat.c needs the
// same declaration).
extern "C" void sam_hdr_destroy(bam_hdr_t *h);

// Singlet: the sorted bins are written straight into the final BAM, in bin order, as each worker finishes
// (direct mode). Upstream STAR wrote every sorted bin to its own BGZF file under _STARtmp/BAMsort and then
// concatenated them with bam_cat: two extra full passes (write + read) of the whole BAM on the node's disk, which
// dominated the sort on deep samples (18-46 min at 112-122M pairs on shared nodes). The records and their order are
// unchanged; only the BGZF block boundaries around the header and between bins differ.
// SINGLET_STAR_LEGACY_SORT=1 restores the upstream path (for A/B checks).
static bool legacySortRequested() {
    const char *e=getenv("SINGLET_STAR_LEGACY_SORT");
    return e!=NULL && e[0]!='\0' && e[0]!='0';
};

// Append the body of a BGZF BAM file (everything after its header, without its EOF block) to fp, like bam_cat.
static void appendBamBody(BGZF *fp, const string &fn, Parameters &P) {
    const int es=28;//BGZF empty (EOF) block
    BGZF *in=bgzf_open(fn.c_str(), "r");
    if (in==NULL) {
        ostringstream errOut;
        errOut <<"EXITING because of fatal ERROR: could not open temporary bam file: " << fn << "\n";
        exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
    };
    bam_hdr_t *old=bam_hdr_read(in);
    if (in->block_offset < in->block_length) {
        bgzf_write(fp, (char*)in->uncompressed_block + in->block_offset, in->block_length - in->block_offset);
        bgzf_flush(fp);
    };
    std::vector<uint8_t> buf(1<<20);
    uint8_t ebuf[es];
    int len, j=0;
    while ((len=bgzf_raw_read(in, buf.data(), buf.size())) > 0) {
        if (len<es) {
            int diff=es-len;
            bgzf_raw_write(fp, ebuf, len);
            memmove(ebuf, ebuf+len, diff);
            memcpy(ebuf+diff, buf.data(), len);
        } else {
            if (j!=0) bgzf_raw_write(fp, ebuf, es);
            len-=es;
            memcpy(ebuf, buf.data()+len, es);
            bgzf_raw_write(fp, buf.data(), len);
        };
        j=1;
    };
    if (old!=NULL) sam_hdr_destroy(old);
    bgzf_close(in);
};

static void sortDirect(Parameters &P, ReadAlignChunk **RAchunk, Genome &genome, Solo &solo, uint32 nBins) {
    typedef std::chrono::steady_clock clk;
    auto secs=[](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b-a).count(); };
    auto t0=clk::now();

    BGZF *bgzfOut=bgzf_open(P.outBAMfileCoordName.c_str(),("w"+to_string((long long) P.outBAMcompression)).c_str());
    if (bgzfOut==NULL) {
        ostringstream errOut;
        errOut <<"EXITING because of fatal ERROR: could not open output bam file: " << P.outBAMfileCoordName << "\n";
        errOut <<"SOLUTION: check that the disk is not full, increase the max number of open files with Linux command ulimit -n before running STAR";
        exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
    };
    outBAMwriteHeader(bgzfOut,P.samHeaderSortedCoord,genome.chrNameAll,genome.chrLengthAll);
    bgzf_flush(bgzfOut);

    const uint32 nMapped=nBins-1;//last bin = unmapped reads
    std::mutex mtx;
    std::vector< std::vector<char> > done(nMapped);
    std::vector<char> ready(nMapped, 0);
    std::vector<uint64> heldMem(nMapped, 0);
    uint32 nextWrite=0;
    bool writing=false;
    uint64 totalMem=0, peakMem=0, bytesOut=0;
    double tSort=0, tWrite=0, tWait=0;
    bool unmappedFile=false;

    #pragma omp parallel num_threads(P.outBAMsortingThreadNactual)
    #pragma omp for schedule (dynamic,1)
    for (uint32 k=0; k<nBins; k++) {
        uint32 ibin = (k==0) ? nBins-1 : k-1;//unmapped bin first (as upstream), then the mapped bins in order
        uint binN=0, binS=0;
        for (int it=0; it<P.runThreadN; it++) {//collect sizes from threads
            binN += RAchunk[it]->chunkOutBAMcoord->binTotalN[ibin];
            binS += RAchunk[it]->chunkOutBAMcoord->binTotalBytes[ibin];
        };
        if (ibin==nBins-1) {
            if (binS>0) {
                BAMbinSortUnmapped(ibin,P.runThreadN,P.outBAMsortTmpDir, P, genome, solo);
                std::lock_guard<std::mutex> lk(mtx);
                unmappedFile=true;
            };
            continue;
        };

        std::vector<char> buf;
        if (binS>0) {
            //wait for RAM; the lowest bin not yet written never waits, so the pipeline always progresses
            uint64 newMem=binS+binN*24;
            auto w0=clk::now();
            while (true) {
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (totalMem+newMem < P.limitBAMsortRAM || ibin==nextWrite) {
                        totalMem+=newMem;
                        if (totalMem>peakMem) peakMem=totalMem;
                        break;
                    };
                };
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            };
            auto w1=clk::now();
            BAMbinSortByCoordinateToBuffer(ibin,binN,binS,P.runThreadN,P.outBAMsortTmpDir, P, genome, solo, buf);
            auto w2=clk::now();
            std::lock_guard<std::mutex> lk(mtx);
            totalMem-=newMem;
            heldMem[ibin]=buf.capacity();
            totalMem+=heldMem[ibin];
            if (totalMem>peakMem) peakMem=totalMem;
            tWait+=secs(w0,w1);
            tSort+=secs(w1,w2);
        };

        //hand the bin to the writer; whoever finds the writer idle writes all consecutive ready bins
        {
            std::lock_guard<std::mutex> lk(mtx);
            done[ibin].swap(buf);
            ready[ibin]=1;
            if (writing) continue;
            writing=true;
        };
        while (true) {
            std::vector<char> out;
            uint32 iw;
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (nextWrite>=nMapped || !ready[nextWrite]) {
                    writing=false;
                    break;
                };
                iw=nextWrite;
                out.swap(done[iw]);
            };
            auto x0=clk::now();
            if (!out.empty() && bgzf_raw_write(bgzfOut, out.data(), out.size()) != (ssize_t) out.size()) {
                ostringstream errOut;
                errOut <<"EXITING because of fatal ERROR: could not write to the output bam file: " << P.outBAMfileCoordName << "\n";
                errOut <<"SOLUTION: check that the disk is not full";
                exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
            };
            auto x1=clk::now();
            std::lock_guard<std::mutex> lk(mtx);
            bytesOut+=out.size();
            totalMem-=heldMem[iw];
            heldMem[iw]=0;
            tWrite+=secs(x0,x1);
            nextWrite++;
        };
    };

    auto t1=clk::now();
    if (unmappedFile)
        appendBamBody(bgzfOut, P.outBAMsortTmpDir+"/b"+to_string((uint) (nBins-1)), P);
    if (bgzf_close(bgzfOut)!=0) {
        ostringstream errOut;
        errOut <<"EXITING because of fatal ERROR: could not close the output bam file: " << P.outBAMfileCoordName << "\n";
        exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
    };
    auto t2=clk::now();
    P.inOut->logMain << "BAM sort (direct): " << nMapped << " bins, " << P.outBAMsortingThreadNactual << " threads, "
                     << bytesOut << " bytes written; wall " << secs(t0,t2) << " s (unmapped append+close " << secs(t1,t2)
                     << " s); thread-seconds: sort+compress " << tSort << ", write " << tWrite << ", RAM wait " << tWait
                     << "; peak sort RAM " << peakMem << " of limit " << P.limitBAMsortRAM << endl;
};

void bamSortByCoordinate (Parameters &P, ReadAlignChunk **RAchunk, Genome &genome, Solo &solo) {
    if (P.outBAMcoord) {//sort BAM if needed
        *P.inOut->logStdOut << timeMonthDayTime() << " ..... started sorting BAM\n" <<flush;
        P.inOut->logMain << timeMonthDayTime() << " ..... started sorting BAM\n" <<flush;
        uint32 nBins=P.outBAMcoordNbins;

        //check max size needed for sorting
        uint maxMem=0;
        for (uint32 ibin=0; ibin<nBins-1; ibin++) {//check all bins
            uint binS=0;
            for (int it=0; it<P.runThreadN; it++) {//collect sizes from threads
                binS += RAchunk[it]->chunkOutBAMcoord->binTotalBytes[ibin]+24*RAchunk[it]->chunkOutBAMcoord->binTotalN[ibin];
            };
            if (binS>maxMem) maxMem=binS;
        };

        uint64 unmappedReadsN = 0;
        for (int it=0; it<P.runThreadN; it++)
            unmappedReadsN += RAchunk[it]->chunkOutBAMcoord->binTotalN[nBins-1];

        P.inOut->logMain << "Max memory needed for sorting = "<<maxMem<<endl;
        if (maxMem>P.limitBAMsortRAM) {
            ostringstream errOut;
            errOut <<"EXITING because of fatal ERROR: not enough memory for BAM sorting: \n";
            errOut <<"SOLUTION: re-run STAR with at least --limitBAMsortRAM " <<maxMem+1000000000;
            exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
        } else if(maxMem==0 && unmappedReadsN==0) {//both mapped and unmapped reads are absent
            P.inOut->logMain << "WARNING: nothing to sort - no output alignments" <<endl;
            BGZF *bgzfOut;
            bgzfOut=bgzf_open(P.outBAMfileCoordName.c_str(),("w"+to_string((long long) P.outBAMcompression)).c_str());
            if (bgzfOut==NULL) {
                ostringstream errOut;
                errOut <<"EXITING because of fatal ERROR: could not open output bam file: " << P.outBAMfileCoordName << "\n";
                errOut <<"SOLUTION: check that the disk is not full, increase the max number of open files with Linux command ulimit -n before running STAR";
                exitWithError(errOut.str(), std::cerr, P.inOut->logMain, EXIT_CODE_PARAMETER, P);
            };
            outBAMwriteHeader(bgzfOut,P.samHeaderSortedCoord,genome.chrNameAll,genome.chrLengthAll);
            bgzf_close(bgzfOut);
        } else if (!legacySortRequested()) {
            sortDirect(P, RAchunk, genome, solo, nBins);
        } else {//sort (upstream path: one BGZF file per bin, then bam_cat)
            uint totalMem=0;
            #pragma omp parallel num_threads(P.outBAMsortingThreadNactual)
            #pragma omp for schedule (dynamic,1)
            for (uint32 ibin1=0; ibin1<nBins; ibin1++) {
                uint32 ibin=nBins-1-ibin1;//reverse order to start with the last bin - unmapped reads

                uint binN=0, binS=0;
                for (int it=0; it<P.runThreadN; it++) {//collect sizes from threads
                    binN += RAchunk[it]->chunkOutBAMcoord->binTotalN[ibin];
                    binS += RAchunk[it]->chunkOutBAMcoord->binTotalBytes[ibin];
                };

                if (binS==0) continue; //empty bin

                if (ibin == nBins-1) {//last bin for unmapped reads
                    BAMbinSortUnmapped(ibin,P.runThreadN,P.outBAMsortTmpDir, P, genome, solo);
                } else {
                    uint newMem=binS+binN*24;
                    bool boolWait=true;
                    while (boolWait) {
                        #pragma omp critical
                        if (totalMem+newMem < P.limitBAMsortRAM) {
                            boolWait=false;
                            totalMem+=newMem;
                        };
                        sleep(0.1);
                    };
                    BAMbinSortByCoordinate(ibin,binN,binS,P.runThreadN,P.outBAMsortTmpDir, P, genome, solo);
                    #pragma omp critical
                    totalMem-=newMem;//"release" RAM
                };
            };

            //concatenate all BAM files, using bam_cat
            char **bamBinNames = new char* [nBins];
            vector <string> bamBinNamesV;
            for (uint32 ibin=0; ibin<nBins; ibin++) {

                bamBinNamesV.push_back(P.outBAMsortTmpDir+"/b"+std::to_string((uint) ibin));
                struct stat buffer;
                if (stat (bamBinNamesV.back().c_str(), &buffer) != 0) {//check if file exists
                    bamBinNamesV.pop_back();
                };
            };
            for (uint32 ibin=0; ibin<bamBinNamesV.size(); ibin++) {
                    bamBinNames[ibin] = (char*) bamBinNamesV.at(ibin).c_str();
            };
            bam_cat(bamBinNamesV.size(), bamBinNames, 0, P.outBAMfileCoordName.c_str());
        };
    };
};
