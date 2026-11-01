#include "StarProfile.h"
#include "ReadAlign.h"
#include "SeqCache.h"
#include "GlobalVariables.h"
#include "ErrorWarning.h"
#include <cstdlib>
#include <mutex>
#include <map>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

//singlet E9 (bench/fable/solo, docs/gpu-engine/fable/E9-solo.md), debug only: SINGLET_SOLO_DUMP=<dir> writes <dir>/solo_<chunk>.bin per
//mapping thread, one 48-byte record per read in the thread's processing order (= the order of its Solo record file), with what Solo's Gene
//feature saw for the read after SoloRead::record: CB and UMI (3 bits per base as handoffReadEntry packs them), the gene set (its gene when
//it has exactly one; its size), the nTr passed to record, STAR's cbMatch / first match index / read flag, the mito and hand-off bits. Unset
//(the default) nothing is written and nothing changes.
struct SoloDumpRec {uint64 iRead, cr3, ur3; uint32 gene, readFlag; int32 cbMatch; uint32 cbInd0; uint16 nTr; uint8 ng, bits; uint32 pad;};
static_assert(sizeof(SoloDumpRec) == 48, "SoloDumpRec layout");
namespace {
struct SoloDumpFile {int fd=-1; std::vector<char> buf;};
std::mutex g_soloDumpMtx;
std::map<int, SoloDumpFile*> g_soloDumpFiles;
const char *soloDumpDir() {static const char *d = getenv("SINGLET_SOLO_DUMP"); return (d != nullptr && d[0] != 0) ? d : nullptr;}
void soloDumpWrite(SoloDumpFile &f) {
    size_t o = 0;
    while (o < f.buf.size()) {ssize_t w = ::write(f.fd, f.buf.data() + o, f.buf.size() - o); if (w <= 0) break; o += (size_t) w;};
    f.buf.clear();
};
uint64 soloDumpPack(const string &s) {
    uint64 v = 0;
    for (size_t j = 0; j < s.size() && j < 21; j++) {
        const char c = s[j];
        v |= (uint64) (c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4) << (3 * j);
    };
    return v;
};
}
void soloDumpFlushAll() {
    std::lock_guard<std::mutex> lk(g_soloDumpMtx);
    for (auto &kv : g_soloDumpFiles) {soloDumpWrite(*kv.second); ::close(kv.second->fd); kv.second->fd = -1;};
};
void ReadAlign::soloDumpRead(uint64 nTrSolo) {
    if (soloDumpDir() == nullptr || P.pSolo.type != P.pSolo.SoloTypes::CB_UMI_Simple || !P.pSolo.featureYes[SoloFeatureTypes::Gene])
        return;
    if (soloDumpF == nullptr) {
        std::lock_guard<std::mutex> lk(g_soloDumpMtx);
        SoloDumpFile *f = new SoloDumpFile;
        f->fd = ::open((string(soloDumpDir()) + "/solo_" + to_string(soloDumpChunk) + ".bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        f->buf.reserve(1 << 22);
        g_soloDumpFiles[soloDumpChunk] = f;
        soloDumpF = f;
    };
    SoloDumpFile &f = *(SoloDumpFile*) soloDumpF;
    if (f.fd < 0) return;
    const ReadAnnotFeature &af = readAnnot.annotFeatures[SoloFeatureTypes::Gene];
    SoloDumpRec r = {};
    r.iRead = iReadAll;
    r.cr3 = soloDumpPack(soloRead->readBar->cbSeq); r.ur3 = soloDumpPack(soloRead->readBar->umiSeq);
    r.gene = af.fSet.size() == 1 ? *af.fSet.begin() : (uint32) -1;
    r.readFlag = soloRead->readFeat[P.pSolo.featureInd[SoloFeatureTypes::Gene]]->readFlag.flag;
    r.cbMatch = soloRead->readBar->cbMatch;
    r.cbInd0 = soloRead->readBar->cbMatchInd.empty() ? (uint32) -1 : (uint32) soloRead->readBar->cbMatchInd[0];
    r.nTr = (uint16) min(nTrSolo, (uint64) 65535);
    r.ng = (uint8) min(af.fSet.size(), (size_t) 255);
    r.bits = (handoffSum != nullptr ? 1 : 0) | (uint8) ((af.ovType & 15) << 4);
    for (uint64 itr = 0; itr < nTrSolo; itr++)
        if (P.pGe.chrSet.mito.count(trMult[itr]->Chr) == 1) r.bits |= 2;
    const char *b = (const char*) &r;
    f.buf.insert(f.buf.end(), b, b + sizeof r);
    if (f.buf.size() >= (1u << 22)) soloDumpWrite(f);
};

void ReadAlign::outputAlignments() {
  
    outBAMbytes=0;
        
    readAnnot.reset();
    
    if (mapGen.pGe.gType==101) {//temporary
        ReadAlign::spliceGraphWriteSAM();
        return;
    };

    if (handoffSum != nullptr) {//singlet GPU pileup "on": a hand-off read replays its sequence's summary (no alignments restored)
        ReadAlign::outputHandoffSummary();
        return;
    };

    ReadAlign::outFilterBySJout();//sets outFilterBySJoutPass=false if read is held for the 2nd stage of outFilterBySJout
    
    if (outFilterBySJoutPass) {//otherwise align is held for the 2nd stage of outFilterBySJout
        ////////////////////////////////////
        PROF_T0(tOutMapped);
        if (unmapType<0) {//passed mappedFilter. Unmapped reads can have nTr>0

            auto nTr1 = nTr;
            auto trOut1 = trMult[0];

            if (P.pGe.transform.outYes) {
                nTr1 = alignsGenOut.alN;
                trOut1 = alignsGenOut.alMult[0];
            };

            if (nTr1>1) {//multimappers
                statsRA.mappedReadsM++;
                unmapType = -2; //not sure if this used
            } else if (nTr1==1) {//unique mappers
                statsRA.mappedReadsU++;
                statsRA.transcriptStats(*trOut1, Lread);
            };

            if (P.pGe.transform.outSAM && (!P.twoPass.yes || P.twoPass.pass2) ) {//transform genome only on 2nd pass
                ReadAlign::recordSJ(alignsGenOut.alN, alignsGenOut.alMult, chunkOutSJ);
            } else {
                ReadAlign::recordSJ(nTr, trMult, chunkOutSJ); //this will set mateMapped
            };            
            
            //singlet --engine gpu: with --soloType CB_samTagOut, no GX/GN/sF attributes, no gene counts and no transcriptomic output,
            //nothing reads the per-read gene annotation (SoloRead::record returns at once for CB_samTagOut), so it is skipped
            static const bool annotUnused = getenv("SINGLET_ENGINE_GPU") != nullptr && getenv("SINGLET_ENGINE_GPU")[0] == '1'
                && P.pSolo.type == P.pSolo.SoloTypes::CB_samTagOut && !P.outSAMattrPresent.GX && !P.outSAMattrPresent.GN
                && !P.outSAMattrPresent.gx && !P.outSAMattrPresent.gn && !P.outSAMattrPresent.sF && !P.quant.geCount.yes && !P.quant.trSAM.yes;
            if (!annotUnused)
                ReadAlign::alignedAnnotation();
        };
        PROF_ADD(PROF_outMapped, tOutMapped);

        //the operations below are both for mapped and unmapped reads
        PROF_T0(tOutBarcode);
        soloRead->readBar->getCBandUMI(Read0, Qual0, readLengthOriginal, readNameExtra[0], readFilesIndex, readName);
        PROF_ADD(PROF_outBarcode, tOutBarcode);
        handoffReadEntry();

        //transcripts: need to be run after CB/UMI are obtained to output CR/UR tags
        if ( P.quant.trSAM.yes && unmapType<0) {//Aligned.toTranscriptome output, only for mapped
            if (P.pGe.transform.outQuant) {
                quantTranscriptome(chunkTr, alignsGenOut.alN,  alignsGenOut.alMult,  alignTrAll);
            } else {
                quantTranscriptome(chunkTr, nTr, trMult,  alignTrAll);
            };
        };        
        
        PROF_T0(tOutSolo);
        soloRead->record((unmapType<0 ? nTr : 0), trMult, iReadAll, readAnnot); //need to supply nTr=0 for unmapped reads
        if (soloDumpDir() != nullptr) soloDumpRead(unmapType<0 ? nTr : 0); //singlet E9 debug dump, off by default
        PROF_ADD(PROF_outSolo, tOutSolo);
        PROF_T0(tOutSAM);

        if (P.pGe.transform.outSAM) {
            ReadAlign::writeSAM(alignsGenOut.alN, alignsGenOut.alMult, alignsGenOut.alBest);
        } else {
            ReadAlign::writeSAM(nTr, trMult, trBest); //this will set mateMapped
        };
        PROF_ADD(PROF_outSAM, tOutSAM);
    };    

    if (unmapType>=0) {//unmapped reads
        statsRA.unmappedAll++; //include unmapType==4, i.e. one-mate alignments of PE reads - which may have been set in writeSAM above
        PROF_T0(tOutUnmapped);
        ReadAlign::outReadsUnmapped(); //uses mateMapped that was set in writeSAM above
        PROF_ADD(PROF_outUnmapped, tOutUnmapped);
    };    
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::recordSJ(uint64 nTrO, Transcript **trO, OutSJ *cSJ)
{//junction output for mapped reads (i.e. passed BySJout filtering)
    if (!P.outSJ.yes)
        return; //no SJ output
    
    if ( P.outSJfilterReads=="All" || nTrO==1 ) {
        uint64 sjReadStartN=cSJ->N;
        for (uint64 iTr=0; iTr<nTrO; iTr++) {//write all transcripts junctions
            outputTranscriptSJ (*(trO[iTr]), nTrO, cSJ, sjReadStartN);
        };
    };
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::outFilterBySJout()
{//filtering by SJout
    outFilterBySJoutPass=true;//only false if the alignment is held for outFilterBySJoutStage. True even if unmapped
    
    if (unmapType>0 || P.outFilterBySJoutStage!=1)
        return; //unmapped, or 2nd stage
   
    for (uint iTr=0;iTr<nTr;iTr++) {//check transcript for unannotated junctions
        for (uint iex=0;iex<trMult[iTr]->nExons-1;iex++) {//check all junctions
            if (trMult[iTr]->canonSJ[iex]>=0 && trMult[iTr]->sjAnnot[iex]==0) {
                outFilterBySJoutPass=false;
                break;
            };
        };
        if (!outFilterBySJoutPass) 
            break;
    };
    
    if (!outFilterBySJoutPass) {//this read is held for further filtering BySJout, record fastq
        unmapType=-3; //the read is not conisdered mapped
        statsRA.readN--;
        statsRA.readBases -= readLength[0]+readLength[1];

        for (uint im=0;im<P.readNends;im++) {
            chunkOutFilterBySJoutFiles[im] << readNameMates[im] <<" "<< iReadAll <<" "<< readFilter <<" "<< readFilesIndex;
            if (!readNameExtra[im].empty())
                chunkOutFilterBySJoutFiles[im]<<" "<< readNameExtra[im];
            chunkOutFilterBySJoutFiles[im] <<"\n";
            chunkOutFilterBySJoutFiles[im] << Read0[im] <<"\n";
            if (readFileType==2) {//fastq
                chunkOutFilterBySJoutFiles[im] << "+\n";
                chunkOutFilterBySJoutFiles[im] << Qual0[im] <<"\n";
            };
        };
    };
    
    //SJ output for all reads, including those not passed bySJout filtering. This only needs to be at the 1st stage of BySJout filtering
    ReadAlign::recordSJ(nTr, trMult, chunkOutSJ1); //this will set mateMapped
         
};

////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::writeSAM(uint64 nTrOutSAM, Transcript **trOutSAM, Transcript *trBestSAM)
{
    outBAMbytes=0;
    mateMapped[0] = mateMapped[1] = false; //mateMapped = are mates present in any of the transcripts?

    if (unmapType < 0 && outFilterBySJoutPass) {//write to SAM/BAM
        
        //////////////////////////////////////////////////////////////////////////////////
        /////////////outSAMfilter
        if (P.outSAMfilter.yes) {
            if (P.outSAMfilter.KeepOnlyAddedReferences) {
                for (uint itr=0;itr<nTrOutSAM;itr++) {//check if transcripts map to chr other than added references
                    if (trOutSAM[itr]->Chr<mapGen.genomeInsertChrIndFirst) {
                        return;//no SAM output
                    };
                };
            } else if (P.outSAMfilter.KeepAllAddedReferences) {
                uint64 nTrOutSAM1=0;
                for (uint itr=0;itr<nTrOutSAM;itr++) {//check if transcripts map to chr other than added references
                    if (trOutSAM[itr]->Chr>=mapGen.genomeInsertChrIndFirst) {
                        trOutSAM[nTrOutSAM1]=trOutSAM[itr];
                        trOutSAM[nTrOutSAM1]->primaryFlag=false;
                        ++nTrOutSAM1;
                    };
                };
                if (nTrOutSAM1==0) {
                   return;//no SAM output
                } else {
                    trOutSAM[0]->primaryFlag=true;
                };
                nTrOutSAM = nTrOutSAM1;
            };
        };
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////// write SAM/BAM 
        auto nTrOutWrite=min(P.outSAMmultNmax,nTrOutSAM); //number of aligns to write to SAM/BAM files            
        
        for (uint iTr=0;iTr<nTrOutWrite;iTr++) {//write transcripts
            //mateMapped1 = true if a mate is present in this transcript
            bool mateMapped1[2]={false,false};
            mateMapped1[trOutSAM[iTr]->exons[0][EX_iFrag]]=true;
            mateMapped1[trOutSAM[iTr]->exons[trOutSAM[iTr]->nExons-1][EX_iFrag]]=true;

            if (P.outSAMbool) {//SAM output
                outBAMbytes+=outputTranscriptSAM(*(trOutSAM[iTr]), nTrOutSAM, iTr, (uint) -1, (uint) -1, 0, -1, NULL, outSAMstream);
                if (P.outSAMunmapped.keepPairs && P.readNmates>1 && ( !mateMapped1[0] || !mateMapped1[1] ) ) {//keep pairs && paired reads && one of the mates not mapped in this transcript //not readNends: this is alignment
                    outBAMbytes+= outputTranscriptSAM(*(trOutSAM[iTr]), 0, 0, (uint) -1, (uint) -1, 0, 4, mateMapped1, outSAMstream);
                };
            };

            // iReadAll is 1-based (processChunks increments it before the read); readRecord is indexed from read 0
            //(a read whose record has no alignments still gets its records, so it reaches the pileup, which stops: STAR and the hand-off disagree)
            const bool handoffCovered = g_handoffReadRecord != nullptr && iReadAll >= 1 && iReadAll - 1 < g_handoffNReads && g_handoffReadRecord[iReadAll - 1] != 0xFFFFFFFFu
                && g_handoffRecAln[g_handoffReadRecord[iReadAll - 1]] > 0;
            if ((P.outBAMunsorted || P.outBAMcoord) && !handoffCovered) {//BAM output (singlet GPU pileup "on": not for hand-off reads)
                alignBAM(*(trOutSAM[iTr]), nTrOutSAM, iTr, mapGen.chrStart[trOutSAM[iTr]->Chr], (uint) -1, (uint) -1, 0, -1, NULL, P.outSAMattrOrder,outBAMoneAlign, outBAMoneAlignNbytes);

                if (P.outBAMunsorted) {//unsorted
                    for (uint imate=0; imate<P.readNmates; imate++) {//output each mate //not readNends: this is alignment
                        outBAMunsorted->unsortedOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], (imate>0 || iTr>0) ? 0 : (outBAMoneAlignNbytes[0]+outBAMoneAlignNbytes[1])*2*nTrOutWrite);
                    };
                    if (P.outSAMunmapped.keepPairs && P.readNmates>1 && ( !mateMapped1[0] || !mateMapped1[1] ) ) {//keep pairs && paired reads && one of the mates not mapped in this transcript //not readNends: this is alignment
                        alignBAM(*trOutSAM[iTr], 0, 0, mapGen.chrStart[trOutSAM[iTr]->Chr], (uint) -1, (uint) -1, 0, 4, mateMapped1, P.outSAMattrOrder, outBAMoneAlign, outBAMoneAlignNbytes);
                        for (uint imate=0; imate<P.readNmates; imate++) {//output each mate //not readNends: this is alignment
                            outBAMunsorted->unsortedOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], (imate>0 || iTr>0) ? 0 : (outBAMoneAlignNbytes[0]+outBAMoneAlignNbytes[1])*2*nTrOutWrite);
                        };
                    };
                };

                if (P.outBAMcoord) {//coordinate sorted
                    for (uint imate=0; imate<P.readNmates; imate++) {//output each mate //not readNends: this is alignment
                        outBAMcoord->coordOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], (iReadAll<<32) | (iTr<<8) | trOutSAM[iTr]->exons[0][EX_iFrag] );
                    };
                };
            };
        };

        /////////////////////////////////////////////////////////////////////////////////////////////
        //////// write unmapped ends
        //TODO it's better to check all transcripts in the loop above for presence of both mates
        mateMapped[trBestSAM->exons[0][EX_iFrag]] = true;
        mateMapped[trBestSAM->exons[trBestSAM->nExons-1][EX_iFrag]] = true;

        if (P.readNmates>1 && !(mateMapped[0] && mateMapped[1]) ) {//not readNends: this is alignment
            unmapType=4;
        };

        if (unmapType==4 && P.outSAMunmapped.within) {//output unmapped ends for single-end alignments of PE reads
            if (P.outSAMbool && !P.outSAMunmapped.keepPairs ) {
                outBAMbytes+= outputTranscriptSAM(*trBestSAM, 0, 0, (uint) -1, (uint) -1, 0, unmapType, mateMapped, outSAMstream);
            };

            if ( P.outBAMcoord || (P.outBAMunsorted && !P.outSAMunmapped.keepPairs) ) {//BAM output
                alignBAM(*trBestSAM, 0, 0, mapGen.chrStart[trBestSAM->Chr], (uint) -1, (uint) -1, 0, unmapType, mateMapped, P.outSAMattrOrder, outBAMoneAlign, outBAMoneAlignNbytes);
                for (uint imate=0; imate<P.readNmates; imate++) {//alignBAM output is empty for mapped mate, but still need to scan through it //not readNends: this is alignment
                    if (P.outBAMunsorted && !P.outSAMunmapped.keepPairs) {
                        outBAMunsorted->unsortedOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], imate>0 ? 0 : outBAMoneAlignNbytes[0]+outBAMoneAlignNbytes[1]);
                    };
                    if (P.outBAMcoord) {//KeepPairs option does not affect for sorted BAM since we do not want multiple entries for the same unmapped read
                        outBAMcoord->coordOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], iReadAll<<32);
                    };
                };
            };
        };  
        
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////write completely unmapped
    } else if (unmapType>=0 && P.outSAMunmapped.within) {//output unmapped within && unmapped read && both mates unmapped
        if (P.outBAMcoord || P.outBAMunsorted || P.quant.trSAM.bamYes) {//BAM output
            alignBAM(*trBestSAM, 0, 0, mapGen.chrStart[trBestSAM->Chr], (uint) -1, (uint) -1, 0, unmapType, mateMapped, P.outSAMattrOrder, outBAMoneAlign, outBAMoneAlignNbytes);
            for (uint imate=0; imate<P.readNmates; imate++) {//output each mate //not readNends: this is alignment
                if (P.outBAMunsorted) {
                    outBAMunsorted->unsortedOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], imate>0 ? 0 : outBAMoneAlignNbytes[0]+outBAMoneAlignNbytes[1]);
                };
                if (P.quant.trSAM.bamYes) {
                    outBAMquant->unsortedOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], imate>0 ? 0 : outBAMoneAlignNbytes[0]+outBAMoneAlignNbytes[1]);
                };
                if (P.outBAMcoord) {
                    outBAMcoord->coordOneAlign(outBAMoneAlign[imate], outBAMoneAlignNbytes[imate], iReadAll<<32);
                };
            };
        };

        if (P.outSAMbool) {//output SAM
            outBAMbytes+= outputTranscriptSAM(*trBestSAM, 0, 0, (uint) -1, (uint) -1, 0, unmapType, mateMapped, outSAMstream);
        };
    };       
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::outReadsUnmapped()
{
    if (P.outReadsUnmapped=="Fastx" ) {//output to fasta/q files. Include unmapType==4, i.e. one-mate alignments of PE reads
       for (uint im=0;im<P.readNends;im++) {
           chunkOutUnmappedReadsStream[im] << readNameMates[im]  <<" "<<im<<":"<< readFilter <<": "<< readNameExtra[im];
           if (P.readNmates>1) //not readNends: this is alignment
               chunkOutUnmappedReadsStream[im] <<" "<< int(mateMapped[0]) <<  int(mateMapped[1]);
           chunkOutUnmappedReadsStream[im] <<"\n";
           chunkOutUnmappedReadsStream[im] << Read0[im] <<"\n";
            if (readFileType==2) {//fastq
                chunkOutUnmappedReadsStream[im] << "+\n";
                chunkOutUnmappedReadsStream[im] << Qual0[im] <<"\n";
            };
       };
    };
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::spliceGraphWriteSAM()
{//temporary: SAM output for SpliceGraph
    outBAMbytes=0;
    uint64 nTrOutSAM = nTr;
    if (mapGen.genomeOut.convYes) {//convert to new genome
        nTrOutSAM=0;
        for (uint32 iTr=0; iTr<nTrOutSAM; iTr++) {//convert output transcripts into new genome
            *alignsGenOut.alMult[nTrOutSAM] = *trMult[iTr];//copy information before conversion
            if (trMult[iTr]->convertGenomeCigar(*mapGen.genomeOut.g, *alignsGenOut.alMult[nTrOutSAM])) {
                ++nTrOutSAM;
                trMult[nTrOutSAM-1] = alignsGenOut.alMult[nTrOutSAM-1]; //point to new transcsript
            };
        };
        nTrOutSAM=nTrOutSAM;
    };

    for (uint iTr=0; iTr<nTrOutSAM; iTr++) {//write all transcripts            
        outBAMbytes += outputSpliceGraphSAM(*(trMult[iTr]), nTrOutSAM, iTr, outSAMstream);
    };
};

void ReadAlign::alignedAnnotation()
{
    //TODO maybe initialize readAnnot to all empty?
    //genes
    if ( P.quant.geCount.yes ) {
        if (P.pGe.transform.outQuant) {
            chunkTr->geneCountsAddAlign(alignsGenOut.alN, alignsGenOut.alMult, readAnnot.geneExonOverlap);
        } else {
            chunkTr->geneCountsAddAlign(nTr, trMult, readAnnot.geneExonOverlap);
        };        
    };
    //solo-GeneFull
    if ( P.quant.geneFull.yes ) {
        chunkTr->geneFullAlignOverlap(nTr, trMult, P.pSolo.strand, readAnnot.annotFeatures[SoloFeatureTypes::GeneFull]);
    };   
    //solo-Gene
    if ( P.quant.gene.yes ) {
        chunkTr->classifyAlign(trMult, nTr, readAnnot);
    };
    //solo-GeneFull_ExonOverIntron
    if ( P.quant.geneFull_ExonOverIntron.yes ) {
        chunkTr->geneFullAlignOverlap_ExonOverIntron(nTr, trMult, P.pSolo.strand, readAnnot.annotFeatures[SoloFeatureTypes::GeneFull_ExonOverIntron], readAnnot.annotFeatures[SoloFeatureTypes::Gene]);
    };
    //solo-GeneFull_Ex50pAS
    if ( P.quant.geneFull_Ex50pAS.yes ) {
        chunkTr->alignExonOverlap(nTr, trMult, P.pSolo.strand, readAnnot.annotFeatures[SoloFeatureTypes::GeneFull_Ex50pAS]);
    };    
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::handoffReadEntry()
{//singlet GPU pileup "on": what the pileup would have taken from this hand-off read's BAM record (CR, UR, base qualities)
    if (g_handoffReads == nullptr || iReadAll < 1 || iReadAll - 1 >= g_handoffNReads || g_handoffReadRecord[iReadAll - 1] == 0xFFFFFFFFu)
        return;
    HandoffReadEntry &e = g_handoffReads[iReadAll - 1];
    auto pack = [](const string &s) {
        uint64 v = 0;
        for (size_t j = 0; j < s.size() && j < 21; j++) {
            const char c = s[j];
            v |= (uint64) (c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4) << (3 * j);
        }
        return v;
    };
    e.cr3 = pack(soloRead->readBar->cbSeq); e.ur3 = pack(soloRead->readBar->umiSeq);
    uint32 lq = 0, ne = 0;
    for (uint64 j = 0; j < readLengthOriginal[0]; j++) {
        const bool low = ((int) (unsigned char) Qual0[0][j] - 33) < g_handoffMinQ, isN = Read0[0][j] == 'N';
        lq += low; ne += (low != isN);
    }
    e.lowQual = (uint16) min(lq, (uint32) 65535); e.qualNeqN = (uint16) min(ne, (uint32) 65535);
    uint32 fl = 1;
    if (g_handoffQMask != nullptr) {//the low-quality mask of the record: the read without its clipped ends, original orientation
        const uint64 c5q = clipMates[0][0].clippedN, c3q = clipMates[0][1].clippedN, Lq = readLengthOriginal[0];
        uint64 *qm = g_handoffQMask + 2 * (iReadAll - 1);
        qm[0] = 0; qm[1] = 0;
        if (c5q + c3q > Lq || Lq - c5q - c3q > 128) {
            fl |= 64;//no mask
        } else {
            for (uint64 j = c5q; j < Lq - c3q; j++)
                if (((int) (unsigned char) Qual0[0][j] - 33) < g_handoffMinQ) qm[(j - c5q) >> 6] |= 1ull << ((j - c5q) & 63);
        };
    };
    //clipped read: its record is the clipped sequence, so the pileup gets the clip lengths and what it reads from the clipped 3' tail of
    //the BAM sequence (poly(A): A count and first base; the rRNA sample: the hook's verdict on the whole read)
    const uint32 c5 = clipMates[0][0].clippedN, c3 = clipMates[0][1].clippedN;
    if (c5 + c3 > 0) {
        uint32 nA = 0;
        for (uint64 j = readLengthOriginal[0] - c3; j < readLengthOriginal[0]; j++) nA += (Read0[0][j] | 32) == 'a';
        if (c5 > 255 || c3 > 255) fl |= 32;
        fl |= (min(c5, 255u) << 8) | (min(c3, 255u) << 16) | (min(nA, 255u) << 24);
        if (c3 > 0 && (Read0[0][readLengthOriginal[0] - c3] | 32) == 'a') fl |= 2;
        if (g_handoffReadHook != nullptr) fl |= g_handoffReadHook(iReadAll - 1, Read0[0], readLengthOriginal[0]) & 0x1Cu;
    }
    e.flags = fl;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ReadAlign::outputHandoffSummary()
{//the mapped-read branch of outputAlignments for a hand-off read, from its sequence's summary (seqCacheHandoffSummary): mapping stats,
 //junctions, the Solo Gene annotation, then barcode and UMI and the Solo record as for any read. No SAM/BAM output (the pileup has the
 //alignments), and the read is mapped (an unmapped sequence has no summary).
    const uint8 *p = handoffSum;
    uint16 n16, ng;
    memcpy(&n16, p, 2);
    const uint8 flags = p[2], ov = p[3];
    memcpy(&ng, p + 4, 2);
    p += 6;
    nTr = n16;
    outFilterBySJoutPass = true;

    PROF_T0(tOutMapped);
    ReadAnnotFeature &af = readAnnot.annotFeatures[SoloFeatureTypes::Gene];
    for (uint16 i = 0; i < ng; i++) {
        uint32 g;
        memcpy(&g, p + 4 * i, 4);
        af.fSet.insert(af.fSet.end(), g);
    };
    p += 4 * ng;
    af.ovType = ov;
    if (nTr > 1) {//multimappers
        statsRA.mappedReadsM++;
        unmapType = -2;
    } else {//unique mappers: Stats::transcriptStats
        statsRA.mappedReadsU++;
        uint16 a[3];
        uint32 b[3];
        memcpy(a, p, 6); memcpy(b, p + 6, 12);
        const uint8 *c = p + 18;
        p += 18 + SJ_MOTIF_SIZE + 1;
        statsRA.mappedMismatchesN += a[0]; statsRA.mappedInsN += a[1]; statsRA.mappedDelN += a[2];
        statsRA.mappedInsL += b[0]; statsRA.mappedDelL += b[1];
        if (flags & 2) {
            for (int k = 0; k < SJ_MOTIF_SIZE; k++) statsRA.splicesN[k] += c[k];
            statsRA.splicesNsjdb += c[SJ_MOTIF_SIZE];
            statsRA.mappedBases += b[2];
            statsRA.mappedPortion += double(b[2]) / double(Lread);
        };
    };
    uint16 nsj;
    memcpy(&nsj, p, 2);
    p += 2;
    if (nsj > 0 && chunkOutSJ != nullptr) {//recordSJ's records for this read
        memcpy(chunkOutSJ->data + chunkOutSJ->N * Junction::dataSize, p, (size_t) nsj * Junction::dataSize);
        chunkOutSJ->N += nsj;
    };
    PROF_ADD(PROF_outMapped, tOutMapped);

    PROF_T0(tOutBarcode);
    soloRead->readBar->getCBandUMI(Read0, Qual0, readLengthOriginal, readNameExtra[0], readFilesIndex, readName);
    PROF_ADD(PROF_outBarcode, tOutBarcode);
    handoffReadEntry();

    PROF_T0(tOutSolo);
    for (int k = 0; k < 2; k++) {
        handoffTr[k].Chr = (k == 0 && !P.pGe.chrSet.mito.empty()) ? *P.pGe.chrSet.mito.begin() : 0;
    };
    if (P.pGe.chrSet.mito.count(handoffTr[1].Chr) == 1) {//a contig that is not mito
        for (uint64 c = 0; c < mapGen.nChrReal; c++) if (P.pGe.chrSet.mito.count(c) == 0) {handoffTr[1].Chr = c; break;};
    };
    for (uint64 i = 0; i < nTr; i++) trMult[i] = &handoffTr[(i == 0 && (flags & 1)) ? 0 : 1];
    trBest = trMult[0];
    soloRead->record(nTr, trMult, iReadAll, readAnnot);
    if (soloDumpDir() != nullptr) soloDumpRead(nTr); //singlet E9 debug dump, off by default
    PROF_ADD(PROF_outSolo, tOutSolo);
};
