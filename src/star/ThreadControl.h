#ifndef THREAD_CONTROL_DEF
#define THREAD_CONTROL_DEF

#include "ReadAlignChunk.h"
#include <pthread.h>
#include "singlet/thread_name.h"   //singlet: per-thread CPU by role (thread names change no output)

#define MAX_chunkOutBAMposition 100000

class ThreadControl {
public:
    bool threadBool;

    pthread_t *threadArray;
    pthread_mutex_t mutexInRead, mutexOutSAM, mutexOutBAM1, mutexOutChimSAM, mutexOutChimJunction, mutexOutUnmappedFastx, mutexOutFilterBySJout;
    pthread_mutex_t mutexStats, mutexLogMain, mutexBAMsortBins, mutexError;

    uint chunkInN,chunkOutN;

    ThreadControl();

    static void* threadRAprocessChunks(void *RAchunk) {
        singlet::set_thread_name("star-map");
        ( (ReadAlignChunk*) RAchunk )->processChunks();
        pthread_exit(0);
        return NULL;
    };
};

#endif

