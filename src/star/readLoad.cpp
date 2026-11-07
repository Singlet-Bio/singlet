#include "readLoad.h"
#include "ErrorWarning.h"
#include "SeqCache.h"

int readLoad(istream& readInStream, Parameters& P, uint& Lread, uint& LreadOriginal, \
            char* readName, char* Seq, char* SeqNum, char* Qual, vector<ClipMate> &clipOneMate, \
            uint &iReadAll, uint32 &readFilesIndex, char &readFilter, string &readNameExtra)
{//load one read from a stream
    int readFileType=0;

    if (readInStream.peek()!='@' && readInStream.peek()!='>') 
        return -1; //end of the stream

    //1st line: readName iReadAll readFilter readFilesIndex [readNameExtra]
    //parsed in place: same result as the former istringstream extraction, without building a stream per read
    static thread_local string line1;
    getline(readInStream, line1); //read 1st line
    const char *lp = line1.c_str();
    auto skipWs = [&lp]() {while (*lp!=0 && isspace((unsigned char) *lp)) lp++;};

    skipWs();
    {
        const char *t = lp;
        while (*lp!=0 && !isspace((unsigned char) *lp)) lp++;
        memcpy(readName, t, lp-t); //TODO check that it does not overflow the array
        readName[lp-t]=0;
    };
    if (strlen(readName)>=DEF_readNameLengthMax-1) {
        ostringstream errOut;
        errOut << "EXITING because of FATAL ERROR in reads input: read name is too long:" << readInStream.gcount()<<"\nRead Name="<<readName<<"\nDEF_readNameLengthMax="<<DEF_readNameLengthMax<<'\n';
        errOut << "SOLUTION: increase DEF_readNameLengthMax in IncludeDefine.h and re-compile STAR\n";
        exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
    };  
    
    {//extract read number TODO this is needed only for one mate
        char *e;
        skipWs();
        iReadAll = strtoull(lp, &e, 10); lp = e;
        skipWs();
        if (*lp!=0) readFilter = *lp++;
        skipWs();
        readFilesIndex = (uint32) strtoul(lp, &e, 10); lp = e;
        skipWs(); //skip whitespace before readNameExtra
        readNameExtra.assign(lp);
    };

    readInStream.getline(Seq,DEF_readSeqLengthMax+1); //extract sequence

    Lread=readInStream.gcount()-1;

    if (Lread<1) {
        ostringstream errOut;
        errOut << "EXITING because of FATAL ERROR in reads input: short read sequence line: " << Lread <<"\n";
        errOut << "Read Name="<< readName <<'\n'<< "Read Sequence=\"" << Seq <<"\"\nDEF_readNameLengthMax="<< DEF_readNameLengthMax <<'\n'<< "DEF_readSeqLengthMax="<<DEF_readSeqLengthMax<<'\n';
        exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
    };
    if (Lread>DEF_readSeqLengthMax) {
        ostringstream errOut;
        errOut << "EXITING because of FATAL ERROR in reads input: Lread>=" << Lread << "   while DEF_readSeqLengthMax=" << DEF_readSeqLengthMax <<'\n'<< "Read Name="<<readName<<'\n';
        errOut << "SOLUTION: increase DEF_readSeqLengthMax in IncludeDefine.h and re-compile STAR\n";
        exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
    };

    LreadOriginal=Lread;

    convertNucleotidesToNumbers(Seq,SeqNum,Lread);
    
    if (readName[0]=='@') {//fastq format: read the '+' line
        readFileType=2;
        readInStream.get(clipOneMate[0].clippedInfo); //this is used if clipChunk is activated, only 5' for now
        if (g_cr4Info != nullptr && iReadAll >= 1 && iReadAll - 1 < g_cr4InfoN)   // singlet --engine gpu: computed by the prefill
            clipOneMate[0].clippedInfo = (char) g_cr4Info[iReadAll - 1];
        readInStream.ignore(); //ignore one char: \n
    };

    //cout << readName <<'\t' << (uint32) line3char << '\n';
    
    //singlet --engine gpu: a read with a hand-off record was clipped on the GPU (engine_gpu.cu k_clip, STAR's ClipMate::clip); its clip
    //lengths replace the CPU clip (SINGLET_HANDOFF_CLIPCHECK=1 runs both and counts differences)
    const bool gpuClip = g_handoffClip5 != nullptr && iReadAll >= 1 && iReadAll - 1 < g_handoffNReads
        && g_handoffReadRecord[iReadAll - 1] != 0xFFFFFFFFu && (clipOneMate[0].type >= 0 || clipOneMate[1].type >= 0);
    if (gpuClip && !g_handoffClipCheck) {
        const uint c5 = g_handoffClip5[iReadAll - 1], c3 = g_handoffClip3[iReadAll - 1];
        clipOneMate[0].clippedN = (uint32) c5; clipOneMate[1].clippedN = (uint32) c3;
        if (c5 > 0) memmove(SeqNum, SeqNum + c5, Lread - c5);
        Lread -= c5 + c3;
    } else {
        clipOneMate[0].clip(Lread, SeqNum); //5p clip
        clipOneMate[1].clip(Lread, SeqNum); //3p clip
        if (gpuClip && (clipOneMate[0].clippedN != g_handoffClip5[iReadAll - 1] || clipOneMate[1].clippedN != g_handoffClip3[iReadAll - 1])
            && g_handoffClipBad++ < 10)
            std::cerr << "[singlet] SINGLET_HANDOFF_CLIPCHECK: read " << iReadAll - 1 << " CPU clip " << clipOneMate[0].clippedN << "/"
                      << clipOneMate[1].clippedN << ", GPU clip " << g_handoffClip5[iReadAll - 1] << "/" << g_handoffClip3[iReadAll - 1] << "\n";
    };

    if (readName[0]=='@') {//fastq format, read qualities
        readInStream.getline(Qual,DEF_readSeqLengthMax);//read qualities
        if ((uint) readInStream.gcount() != LreadOriginal+1) {//inconsistent read sequence and quality
            ostringstream errOut;
            errOut << "EXITING because of FATAL ERROR in reads input: quality string length is not equal to sequence length\n";
            errOut << readName <<'\n'<< Seq <<'\n'<< Qual <<'\n'<< "SOLUTION: fix your fastq file\n";
            exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
        };
        if (P.outQSconversionAdd!=0) {
            for (uint ii=0;ii<LreadOriginal;ii++) {
                int qs=int(Qual[ii])+P.outQSconversionAdd;
                if (qs<33) {
                    qs=33;
                } else if (qs>126) {
                    qs=126;
                };
                Qual[ii]=qs;
            };
        };
    
    } else if (readName[0]=='>') {//fasta format, assign Qtop to all qualities
        readFileType=1;
        for (uint ii=0;ii<LreadOriginal;ii++) 
            Qual[ii]='A';
        Qual[LreadOriginal]=0;
    } else {//wrong header
        ostringstream errOut;
        errOut <<"Unknown reads file format: header line does not start with @ or > : "<< readName<<"\n";
        exitWithError(errOut.str(),std::cerr, P.inOut->logMain, EXIT_CODE_INPUT_FILES, P);
    };

    //trim read name TODO this is needed only for one mate
    for (uint ii=0; ii<P.readNameSeparatorChar.size(); ii++) {
        char* pSlash=strchr(readName,P.readNameSeparatorChar.at(ii)); //trim everything after ' '
        if (pSlash!=NULL) *pSlash=0;
    };
    return readFileType;
};
