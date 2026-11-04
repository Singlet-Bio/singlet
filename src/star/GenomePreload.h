// singlet E14: the genome index read once per process tree (GenomePreload.cpp; the public half is in star_api.h).
#pragma once
#include <cstdint>
#include <ctime>
#include <string>
#include <sys/types.h>

struct GenomePreloadState {
    std::string dir;
    dev_t dev = 0;
    ino_t ino = 0;            // of the index directory
    time_t mtime[3] = {0, 0, 0};   // Genome, SA, SAindex
    uint64_t nGenome = 0, nSAbyte = 0, saiHdr = 0, saiBytes = 0;
    char *G1 = nullptr, *SA = nullptr, *SAi = nullptr;   // new[] arrays: G1 = 200 + nGenome + 200, SA = nSAbyte + 64, SAi = saiBytes + 64
    double seconds = 0;
    bool resident = false;    // E21: read by `singlet gpu-pool` for the processes it forks (star_genome_preload_resident)
};
extern GenomePreloadState* g_genomePreload;   // this process's copy (a forked child has its own pointer to the shared pages)

// Genome::genomeLoad (NoSharedMemory, no sequence or junction insertion): the preloaded arrays when the directory and every
// size match STAR's own (then this process owns them: Genome::freeMemory deletes them), else false and the reason in log.
bool star_genome_preload_take(const std::string& gDir, uint64_t nGenome, uint64_t nSAbyte, uint64_t saLengthByte,
                              uint64_t saiHdr, uint64_t saiLengthByte, char*& G1, char*& SA, char*& SAi, std::string& log);
void star_genome_preload_release();   // also in star_api.h: frees this process's references (a STAR run that does not take them)
