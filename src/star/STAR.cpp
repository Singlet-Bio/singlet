#include "StarProfile.h"
#include <chrono>
#ifdef SINGLET_PGO_GENERATE
extern "C" void __gcov_dump(void);
#endif
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#endif

#include "IncludeDefine.h"
#include "Parameters.h"
#include "SequenceFuns.h"
#include "Genome.h"
#include "Chain.h"
#include "ReadAlignChunk.h"
#include "ReadAlign.h"
#include "SeqCache.h"
void seqCacheGpuPrefill(Parameters &P, Genome &G, SeqCache *&cache, Transcriptome *tr);
void seqCacheGpuAfterMapping(Parameters &P, ReadAlignChunk **RAchunk); //singlet E10 (SeqCacheGpuPrefill.cpp)
void seqCacheGpuEarly(Parameters &P); //singlet E13: the decoded reads' barcode pass (and clip) for the prefill, beside the genome load (SeqCacheGpuPrefill.cpp)
extern bool g_e11SoloDone; //singlet E11 (SeqCacheGpuPrefill.cpp): star_Solo.out/ was written from the device's counts
#include "StarAccel.h"
#include "Stats.h"
#include "genomeGenerate.h"
#include "outputSJ.h"
#include "ThreadControl.h"
#include "GlobalVariables.h"
#include "TimeFunctions.h"
#include "ErrorWarning.h"
#include "sysRemoveDir.h"
#include "BAMfunctions.h"
#include "bamSortByCoordinate.h"
#include "Transcriptome.h"
#include "signalFromBAM.h"
#include "mapThreadsSpawn.h"
#include "SjdbClass.h"
#include "sjdbInsertJunctions.h"
#include "Variation.h"
#include "Solo.h"
#include "samHeaders.h"
#include "systemFunctions.h"

#include "twoPassRunPass1.h"

#include "htslib/htslib/sam.h"
#include "parametersDefault.xxd"

void usage(int usageType)
{
    cout << "Usage: STAR  [options]... --genomeDir /path/to/genome/index/   --readFilesIn R1.fq R2.fq\n";
    cout << "Spliced Transcripts Alignment to a Reference\n";
    cout << "Originally by Alexander Dobin (MIT license). Bundled in singlet.\n\n";
    cout << "STAR version=" << STAR_VERSION << "\n";
    cout << "STAR compilation time,server,dir=" << COMPILATION_TIME_PLACE << "\n";

    if (usageType == 0)
    { // brief
        cout << "\nTo list all parameters, run STAR --help\n";
    }
    else if (usageType == 1)
    { // full
        cout.write(reinterpret_cast<char *>(parametersDefault),
                   parametersDefault_len);
    };
    exit(0);
};

// Entry point callable from the singlet pipeline (replaces standalone main).
// Renamed from main() so the singlet binary links this translation unit without
// duplicate-symbol conflicts.
int star_main_impl(int argInN, char *argIn[])
{
    // If no argument is given, or the first argument is either '-h' or '--help', run usage()
    if (argInN == 1)
    {
        usage(0);
    }
    else if (argInN == 2 && (strcmp("-h", argIn[1]) == 0 || strcmp("--help", argIn[1]) == 0))
    {
        usage(1);
    };

    time(&g_statsAll.timeStart);

    // On multi-socket (NUMA) systems, interleave memory across all nodes.
    // STAR's working set (genome + SA ≈ 13GB) far exceeds any single node's
    // local memory bandwidth. Interleaving distributes pages round-robin,
    // averaging remote/local access latency. Benchmarked: ~10% wall-time
    // improvement on 2-socket Xeon Gold (4 NUMA nodes). No effect on
    // single-socket systems.
#ifdef __linux__
    {
        unsigned long nodemask = ~0UL;
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8);
    }
#endif

    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////// Parameters
    Parameters P; // all parameters
    P.inputParameters(argInN, argIn);

    *(P.inOut->logStdOut) << "\t" << P.commandLine << '\n';
    *(P.inOut->logStdOut) << "\tSTAR version: " << STAR_VERSION << "   (singlet-bundled)   compiled: " << COMPILATION_TIME_PLACE << '\n';
    *(P.inOut->logStdOut) << timeMonthDayTime(g_statsAll.timeStart) << " ..... started STAR run\n"
                          << flush;

    // runMode
    if (P.runMode == "alignReads" || P.runMode == "soloCellFiltering")
    {
        // continue
    }
    else if (P.runMode == "genomeGenerate")
    {
        { // normal genome generation
            Genome genomeMain(P, P.pGe);
            genomeMain.genomeGenerate();
        };

        if (P.pGe.transform.type > 0)
        { // generate original genome, in addition to the transfomed generated above
            P.pGe.transform.type = 0;
            P.pGe.transform.typeString = "None";
            P.pGe.transform.vcfFile = "-";
            P.pGe.gDir += "/OriginalGenome/";
            Genome genomeOrig(P, P.pGe);
            genomeOrig.genomeGenerate();
        };

        sysRemoveDir(P.outFileTmp);
        P.inOut->logMain << "DONE: Genome generation, EXITING\n"
                         << flush;
        exit(0);
    }
    else if (P.runMode == "liftOver")
    {
        for (uint ii = 0; ii < P.pGe.gChainFiles.size(); ii++)
        {
            Chain chain(P, P.pGe.gChainFiles.at(ii));
            chain.liftOverGTF(P.pGe.sjdbGTFfile, P.outFileNamePrefix + "GTFliftOver_" + to_string(ii + 1) + ".gtf");
            P.inOut->logMain << "DONE: lift-over of GTF file, EXITING\n"
                             << flush;
            exit(0);
        };
    }
    else
    {
        P.inOut->logMain << "EXITING because of INPUT ERROR: unknown value of input parameter runMode=" << P.runMode << endl
                         << flush;
        exit(1);
    };

    // transcripome placeholder
    Transcriptome *transcriptomeMain = NULL;

    // this will execute --runMode soloCellFiltering and exit
    Solo soloCellFilter(P, *transcriptomeMain);

    seqCacheGpuEarly(P); //singlet --engine gpu, E13 (SINGLET_ENGINE_GPU_INGEST=1): the reads' barcodes and clip on a thread while the genome loads

    ////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// Genome
    Genome genomeMain(P, P.pGe);
    genomeMain.genomeLoad();

    if (P.pGe.transform.outYes) {
        genomeMain.Var = new Variation(P, genomeMain.chrStart, genomeMain.chrNameIndex, false);//no variation for mapGen, only for genOut
        genomeMain.genomeOut.g->Var = new Variation(P, genomeMain.genomeOut.g->chrStart, genomeMain.genomeOut.g->chrNameIndex, P.var.yes);
    } else {
        genomeMain.Var = new Variation(P, genomeMain.chrStart, genomeMain.chrNameIndex, P.var.yes);
    };

    SjdbClass sjdbLoci;

    if (P.sjdbInsert.pass1) {
        Genome genomeMain1 = genomeMain; // not sure if I need to create the copy - genomeMain1 below should not be changed
        sjdbInsertJunctions(P, genomeMain, genomeMain1, sjdbLoci);
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////START
    if (P.runThreadN > 1)
    {
        g_threadChunks.threadArray = new pthread_t[P.runThreadN];
        pthread_mutex_init(&g_threadChunks.mutexInRead, NULL);
        pthread_mutex_init(&g_threadChunks.mutexOutSAM, NULL);
        pthread_mutex_init(&g_threadChunks.mutexOutBAM1, NULL);
        pthread_mutex_init(&g_threadChunks.mutexOutUnmappedFastx, NULL);
        pthread_mutex_init(&g_threadChunks.mutexOutFilterBySJout, NULL);
        pthread_mutex_init(&g_threadChunks.mutexStats, NULL);
        pthread_mutex_init(&g_threadChunks.mutexBAMsortBins, NULL);
        pthread_mutex_init(&g_threadChunks.mutexError, NULL);
    };

    g_statsAll.progressReportHeader(P.inOut->logProgress);

    /////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////// 2-pass 1st pass
    twoPassRunPass1(P, genomeMain, transcriptomeMain, sjdbLoci);

    if (P.quant.yes)
    { // load transcriptome
        transcriptomeMain = new Transcriptome(P);
    };

    // initialize Stats
    g_statsAll.resetN();
    time(&g_statsAll.timeStartMap);
    *P.inOut->logStdOut << timeMonthDayTime(g_statsAll.timeStartMap) << " ..... started mapping\n"
                        << flush;
    g_statsAll.timeLastReport = g_statsAll.timeStartMap;

    // SAM headers
    samHeaders(P, *genomeMain.genomeOut.g, *transcriptomeMain);

    // initialize chimeric parameters here - note that chimeric parameters require samHeader
    P.pCh.initialize(&P);

    // this does not seem to work at the moment
    // P.inOut->logMain << "mlock value="<<mlockall(MCL_CURRENT|MCL_FUTURE) <<"\n"<<flush;

    // sequence cache for the main mapping pass (after any 2-pass junction insertion)
    g_seqCache = SeqCache::create(P, genomeMain.nGenome);
    seqCacheGpuPrefill(P, genomeMain, g_seqCache, transcriptomeMain); //singlet --engine gpu: GPU mapping results into the cache (SeqCacheGpuPrefill.cpp)
    {
        const char *fa = getenv("SINGLET_STAR_FASTASSIGN");
        g_starFastAssign = !(fa != nullptr && fa[0] == '0');
        P.inOut->logMain << "FASTASSIGN: " << (g_starFastAssign ? "on" : "off") << "\n";
    }
#ifdef STAR_PROFILE
    auto profClock0 = std::chrono::steady_clock::now(); unsigned long long profTsc0 = __rdtsc();
#endif

    // prepare chunks and spawn mapping threads
    ReadAlignChunk *RAchunk[P.runThreadN];
    for (int ii = 0; ii < P.runThreadN; ii++)
    {
        RAchunk[ii] = new ReadAlignChunk(P, genomeMain, transcriptomeMain, ii);
    };

    if (P.runRestart.type != 1)
        mapThreadsSpawn(P, RAchunk);

    if (P.outFilterBySJoutStage == 1)
    { // completed stage 1, go to stage 2
        P.inOut->logMain << "Completed stage 1 mapping of outFilterBySJout mapping\n"
                         << flush;
        outputSJ(RAchunk, P); // collapse novel junctions
        P.readFilesIndex = -1;

        P.outFilterBySJoutStage = 2;
        if (P.outBAMcoord)
        {
            for (int it = 0; it < P.runThreadN; it++)
            { // prepare the unmapped bin
                RAchunk[it]->chunkOutBAMcoord->coordUnmappedPrepareBySJout();
            };
        };

        mapThreadsSpawn(P, RAchunk);
    };

    // close some BAM files
    if (P.inOut->outBAMfileUnsorted != NULL)
    {
        bgzf_flush(P.inOut->outBAMfileUnsorted);
        bgzf_close(P.inOut->outBAMfileUnsorted);
    };
    if (P.inOut->outQuantBAMfile != NULL)
    {
        bgzf_flush(P.inOut->outQuantBAMfile);
        bgzf_close(P.inOut->outQuantBAMfile);
    };

    if (P.outBAMcoord && P.limitBAMsortRAM == 0)
    { // make it equal ot the genome size
        P.limitBAMsortRAM = genomeMain.nGenome + genomeMain.SA.lengthByte + genomeMain.SAi.lengthByte;
    };

    time(&g_statsAll.timeFinishMap);
    *P.inOut->logStdOut << timeMonthDayTime(g_statsAll.timeFinishMap) << " ..... finished mapping\n"
                        << flush;
    P.inOut->logMain << timeMonthDayTime(g_statsAll.timeFinishMap) << " ..... finished mapping\n"
                     << "RAM after mapping:\n"
                     << linuxProcMemory() << flush;
    seqCacheGpuAfterMapping(P, RAchunk); //singlet --engine gpu, E10 (SINGLET_ENGINE_GPU_E2E >= 2); nothing otherwise
#ifdef STAR_PROFILE
    {
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - profClock0).count();
        double hz = (__rdtsc() - profTsc0) / sec;
        const char *nm[PROF_N] = {"chunkRead", "chunkMutexWait", "chunkSort", "readLoad", "map", "output", "chunkWrite", "seed", "windows", "stitch",
                                   "outMapped", "outBarcode", "outSolo", "outSAM", "outUnmapped", "outEntry"};
        P.inOut->logMain << "STAR_PROFILE wall=" << sec << "s threads=" << P.runThreadN;
        for (int i = 0; i < PROF_N; i++)
            P.inOut->logMain << " " << nm[i] << "=" << g_prof[i] / hz << "s";
        P.inOut->logMain << " pathSummary=" << g_profPath[0] << " pathCacheHit=" << g_profPath[1] << " pathMapped=" << g_profPath[2];
        const char *on[6] = {"unique", "multi", "tooManyLoci", "tooShort", "otherUnmapped", "cacheHit"};
        for (int i = 0; i < 6; i++)
            P.inOut->logMain << " " << on[i] << "=" << g_profOutcome[i][1] << "reads/" << g_profOutcome[i][0] / hz << "s"
                             << "[seed=" << g_profOutcomeDetail[i][0] / hz << ",win=" << g_profOutcomeDetail[i][1] / hz << ",stitch=" << g_profOutcomeDetail[i][2] / hz
                             << ",mmp=" << g_profOutcomeDetail[i][3] << ",saSearch=" << g_profOutcomeDetail[i][4] << ",windows=" << g_profOutcomeDetail[i][5]
                             << ",shortProven=" << g_profOutcomeDetail[i][6]
                             << ",winCreate=" << g_profOutcomeDetail[i][7] / hz << ",winFlank=" << g_profOutcomeDetail[i][8] / hz
                             << ",winAssign=" << g_profOutcomeDetail[i][9] / hz << ",saAnchor=" << g_profOutcomeDetail[i][10]
                             << ",saAll=" << g_profOutcomeDetail[i][11] << "]";
        const char *cn[7] = {"g1perfectU", "g1perfectS", "g1mm1U", "g1mm1S", "g1otherU", "g1otherS", "uniqueOther"};
        for (int i = 0; i < 7; i++)
            P.inOut->logMain << " " << cn[i] << "=" << g_profClass[i][1] << "reads/" << g_profClass[i][0] / hz << "s"
                             << "[seed=" << g_profClassDetail[i][0] / hz << ",win=" << g_profClassDetail[i][1] / hz << ",stitch=" << g_profClassDetail[i][2] / hz
                             << ",mmp=" << g_profClassDetail[i][3] << ",saSearch=" << g_profClassDetail[i][4] << "]";
        P.inOut->logMain << "\n" << flush;
    }
#endif
    if (g_seqCache != nullptr)
    {
        P.inOut->logMain << g_seqCache->stats() << "\n" << flush;
        delete g_seqCache;
        g_seqCache = nullptr;
    };

    // no need for genome anymore, free the memory
    genomeMain.freeMemory();

    // aggregate output junctions
    // collapse splice junctions from different threads/chunks, and output them
    if (P.runRestart.type != 1 && P.outSJ.yes)
        outputSJ(RAchunk, P);

    soloDumpFlushAll(); // singlet E9 debug dump (SINGLET_SOLO_DUMP, off by default): the threads' per-read Solo records

    // solo counts
    Solo soloMain(RAchunk, P, *RAchunk[0]->chunkTr);
    if (!g_e11SoloDone) //singlet E11 (SINGLET_ENGINE_GPU_E2E=5): counted on the device
    soloMain.processAndOutput();

    if (P.quant.geCount.yes)
    { // output gene quantifications
        for (int ichunk = 1; ichunk < P.runThreadN; ichunk++)
        { // sum counts from all chunks into 0th chunk
            RAchunk[0]->chunkTr->quants->addQuants(*(RAchunk[ichunk]->chunkTr->quants));
        };
        RAchunk[0]->chunkTr->quantsOutput();
    };

    if (P.runThreadN > 1 && P.outSAMorder == "PairedKeepInputOrder")
    { // concatenate Aligned.* files
        RAchunk[0]->chunkFilesCat(P.inOut->outSAM, P.outFileTmp + "/Aligned.out.sam.chunk", g_threadChunks.chunkOutN);
    };

    bamSortByCoordinate(P, RAchunk, *genomeMain.genomeOut.g, soloMain);

    // wiggle output
    if (P.outWigFlags.yes)
    {
        *(P.inOut->logStdOut) << timeMonthDayTime() << " ..... started wiggle output\n"
                              << flush;
        P.inOut->logMain << timeMonthDayTime() << " ..... started wiggle output\n"
                         << flush;
        string wigOutFileNamePrefix = P.outFileNamePrefix + "Signal";
        signalFromBAM(P.outBAMfileCoordName, wigOutFileNamePrefix, P);
    };

    g_statsAll.writeLines(P.inOut->outChimJunction, P.pCh.outJunctionFormat, "#", STAR_VERSION + string("   ") + P.commandLine);

    g_statsAll.progressReport(P.inOut->logProgress);
    P.inOut->logProgress << "ALL DONE!\n"
                         << flush;
    P.inOut->logFinal.open((P.outFileNamePrefix + "Log.final.out").c_str());
    g_statsAll.reportFinal(P.inOut->logFinal);
    *P.inOut->logStdOut << timeMonthDayTime(g_statsAll.timeFinish) << " ..... finished successfully\n"
                        << flush;

    P.inOut->logMain << "ALL DONE!\n"
                     << flush;
    if (P.outTmpKeep == "None")
    {
        sysRemoveDir(P.outFileTmp);
    };

    P.closeReadsFiles(); // this will kill the readFilesCommand processes if necessary
    // genomeMain.~Genome(); //need explicit call because of the 'delete P.inOut' below, which will destroy P.inOut->logStdOut
    if (genomeMain.sharedMemory != NULL)
    { // need explicit call because this destructor will write to files which are deleted by 'delete P.inOut' below
        delete genomeMain.sharedMemory;
        genomeMain.sharedMemory = NULL;
    };

    delete P.inOut; // to close files

#ifdef SINGLET_PGO_GENERATE
    // profile-generate builds (src/star/pgo/pgo_build.sh): the pipeline's STAR child leaves with
    // _exit(), which skips the gcov atexit handler, so write the profile here
    __gcov_dump();
#endif

    return 0;
};
