// singlet E14 (docs/gpu-engine/fable/E14-probe.md): the genome index read once per process tree.
//
// star_genome_preload() reads Genome, SA and SAindex of a STAR index directory into private memory with several reader
// threads, in exactly the layout Genome::genomeLoad (NoSharedMemory) builds: G1 = 200 padding bytes (value K-1 = 5),
// the genome, 200 padding bytes; SA and the SAindex data (after its header) as new[] arrays, transparent hugepages
// advised before the first touch (SA_HUGEPAGES, as STAR's own load). Genome::genomeLoad (Genome_genomeLoad.cpp) takes
// these arrays instead of reading the files when the directory and every size match (star_genome_preload_take), so a
// process that forks STAR after the load, and every STAR process forked from that one (the CR4 probes, the main run),
// shares the pages copy-on-write: STAR never writes into G, SA or SAi. The bytes are the files' bytes, so the
// alignments are the same as from STAR's own load.
#include "star_api.h"
#include "GenomePreload.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

GenomePreloadState* g_genomePreload = nullptr;

namespace {
const uint64_t kPad = 200;   // STAR's L (Genome_genomeLoad.cpp)
const char kPadVal = 5;      // STAR's K-1
const uint64_t kTail = 64;   // zero bytes after SA and SAi (STAR's PackedArray reads up to 8 bytes past the last word)

void advise(char* p, uint64_t len) {
#ifdef SA_HUGEPAGES
    const uintptr_t pg = 4096;
    uintptr_t a = ((uintptr_t) p + pg - 1) & ~(pg - 1), e = ((uintptr_t) p + len) & ~(pg - 1);
    if (e > a) madvise((void*) a, e - a, MADV_HUGEPAGE);
#else
    (void) p; (void) len;
#endif
}

struct Piece { int fd; uint64_t off; char* dst; uint64_t n; };

bool readPieces(const std::vector<Piece>& pieces, int threads) {
    std::atomic<size_t> next{0};
    std::atomic<bool> ok{true};
    auto work = [&]() {
        for (size_t i; ok && (i = next++) < pieces.size();) {
            const Piece& p = pieces[i];
            uint64_t done = 0;
            while (done < p.n) {
                ssize_t r = pread(p.fd, p.dst + done, p.n - done, (off_t) (p.off + done));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) { ok = false; break; }
                done += (uint64_t) r;
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < threads; ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
    return ok;
}

uint64_t anonHugeKB() {   // AnonHugePages of this process (smaps_rollup), for the log
    std::ifstream f("/proc/self/smaps_rollup");
    std::string k; uint64_t v = 0;
    while (f >> k) {
        if (k == "AnonHugePages:") { f >> v; return v; }
        f.ignore(1 << 16, '\n');
    }
    return 0;
}
}  // namespace

bool star_genome_preload(const std::string& dir, int threads, std::string& log) {
    if (g_genomePreload != nullptr) {
        struct stat sd{};
        if (stat(dir.c_str(), &sd) == 0 && sd.st_dev == g_genomePreload->dev && sd.st_ino == g_genomePreload->ino) {
            char b[160];
            snprintf(b, sizeof b, "already in memory (read in %.2f s, before this process was forked%s)", g_genomePreload->seconds,
                     g_genomePreload->resident ? ", by the gpu-pool process" : "");
            log = b;
            return true;
        }
        star_genome_preload_release();   // E21: another index directory (a gpu-pool worker's sample of another genome): read this one
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto* S = new GenomePreloadState;
    S->dir = dir;
    struct stat sd{};
    if (stat(dir.c_str(), &sd) != 0) { log = "no index directory " + dir; delete S; return false; }
    S->dev = sd.st_dev; S->ino = sd.st_ino;
    int fd[3] = {-1, -1, -1};
    const char* names[3] = {"Genome", "SA", "SAindex"};
    uint64_t size[3] = {0, 0, 0};
    auto fail = [&](const std::string& why) {
        for (int i = 0; i < 3; ++i) if (fd[i] >= 0) close(fd[i]);
        delete[] S->G1; delete[] S->SA; delete[] S->SAi; delete S;
        log = why;
        return false;
    };
    for (int i = 0; i < 3; ++i) {
        fd[i] = open((dir + "/" + names[i]).c_str(), O_RDONLY);
        struct stat st{};
        if (fd[i] < 0 || fstat(fd[i], &st) != 0 || st.st_size <= 0) return fail(std::string("cannot open ") + names[i]);
        size[i] = (uint64_t) st.st_size;
        S->mtime[i] = st.st_mtime;
    }
    uint64_t saiN = 0;   // SAindex: gSAindexNbases (8 bytes), genomeSAindexStart[gSAindexNbases + 1], then the array
    if (pread(fd[2], &saiN, 8, 0) != 8 || saiN == 0 || saiN > 64) return fail("SAindex header");
    S->saiHdr = 8 + 8 * (saiN + 1);
    if (size[2] <= S->saiHdr) return fail("SAindex size");
    S->nGenome = size[0];
    S->nSAbyte = size[1];
    S->saiBytes = size[2] - S->saiHdr;
    try {
        S->G1 = new char[S->nGenome + 2 * kPad];
        S->SA = new char[S->nSAbyte + kTail];
        S->SAi = new char[S->saiBytes + kTail];
    } catch (const std::exception&) {
        return fail("cannot allocate " + std::to_string((S->nGenome + S->nSAbyte + S->saiBytes) >> 20) + " MB");
    }
    advise(S->G1, S->nGenome + 2 * kPad);
    advise(S->SA, S->nSAbyte + kTail);
    advise(S->SAi, S->saiBytes + kTail);
    std::vector<Piece> pieces;
    const uint64_t piece = 64ull << 20;
    auto add = [&](int f, uint64_t off, char* dst, uint64_t n) {
        for (uint64_t o = 0; o < n; o += piece) pieces.push_back({f, off + o, dst + o, std::min(piece, n - o)});
    };
    add(fd[1], 0, S->SA, S->nSAbyte);   // largest first
    add(fd[0], 0, S->G1 + kPad, S->nGenome);
    add(fd[2], S->saiHdr, S->SAi, S->saiBytes);
    const bool ok = readPieces(pieces, std::max(1, threads));
    for (int i = 0; i < 3; ++i) { close(fd[i]); fd[i] = -1; }
    if (!ok) return fail("read error");
    memset(S->G1, kPadVal, kPad);
    memset(S->G1 + kPad + S->nGenome, kPadVal, kPad);
    memset(S->SA + S->nSAbyte, 0, kTail);
    memset(S->SAi + S->saiBytes, 0, kTail);
    S->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    g_genomePreload = S;
    char b[256];
    snprintf(b, sizeof b, "%.0f MB in %.2f s (%d threads; Genome %llu, SA %llu, SAindex %llu bytes; AnonHugePages %.1f GB)",
             (double) (S->nGenome + S->nSAbyte + S->saiBytes) / (1 << 20), S->seconds, std::max(1, threads),
             (unsigned long long) S->nGenome, (unsigned long long) S->nSAbyte, (unsigned long long) size[2],
             (double) anonHugeKB() / (1 << 20));
    log = b;
    return true;
}

// E21 (docs/gpu-engine/fable/E21-pool.md): `singlet gpu-pool` reads each genome of its queue once, before it forks the samples' processes,
// and keeps it for its whole life (star_genome_preload_resident). A forked sample process selects the one for its --genome-dir
// (star_genome_preload_select): from then on it is that process's preload, exactly as if the process had read it itself (the STAR child
// takes it, the process drops its references after the STAR fork); the pages stay shared copy-on-write with the pool and every
// other sample. The pool itself never maps STAR or touches the arrays again.
namespace { std::vector<GenomePreloadState*>* g_resident = nullptr; }

bool star_genome_preload_resident(const std::string& dir, int threads, std::string& log) {
    GenomePreloadState* own = g_genomePreload;
    g_genomePreload = nullptr;
    const bool ok = star_genome_preload(dir, threads, log);
    if (ok) {
        if (g_resident == nullptr) g_resident = new std::vector<GenomePreloadState*>;
        g_genomePreload->resident = true;
        g_resident->push_back(g_genomePreload);
    }
    g_genomePreload = own;
    return ok;
}

bool star_genome_preload_select(const std::string& dir, std::string& log) {
    struct stat sd{};
    if (g_resident == nullptr || g_resident->empty()) { log = "no resident genome"; return false; }
    if (stat(dir.c_str(), &sd) != 0) { log = "no index directory " + dir; return false; }
    for (GenomePreloadState* S : *g_resident)
        if (S->dev == sd.st_dev && S->ino == sd.st_ino) {
            if (g_genomePreload != nullptr && g_genomePreload != S) star_genome_preload_release();
            g_genomePreload = new GenomePreloadState(*S);   // this process's state over the shared arrays (take/release delete the state)
            char b[160];
            snprintf(b, sizeof b, "resident in the gpu-pool process (%.0f MB, read in %.2f s)",
                     (double) (S->nGenome + S->nSAbyte + S->saiBytes) / (1 << 20), S->seconds);
            log = b;
            return true;
        }
    log = "not resident (the pool holds " + std::to_string(g_resident->size()) + " other genome(s))";
    return false;
}

void star_genome_preload_release() {
    if (g_genomePreload == nullptr) return;
    delete[] g_genomePreload->G1;
    delete[] g_genomePreload->SA;
    delete[] g_genomePreload->SAi;
    delete g_genomePreload;
    g_genomePreload = nullptr;
}

bool star_genome_preload_take(const std::string& gDir, uint64_t nGenome, uint64_t nSAbyte, uint64_t saLengthByte,
                              uint64_t saiHdr, uint64_t saiLengthByte, char*& G1, char*& SA, char*& SAi, std::string& log) {
    GenomePreloadState* S = g_genomePreload;
    if (S == nullptr) return false;
    struct stat sd{};
    std::string why;
    if (stat(gDir.c_str(), &sd) != 0 || sd.st_dev != S->dev || sd.st_ino != S->ino) why = "another index directory";
    else if (nGenome != S->nGenome) why = "Genome size";
    else if (nSAbyte != S->nSAbyte || saLengthByte > S->nSAbyte + kTail - 8) why = "SA size";
    else if (saiHdr != S->saiHdr || saiLengthByte > S->saiBytes + kTail - 8) why = "SAindex size";
    else {
        const char* names[3] = {"Genome", "SA", "SAindex"};
        for (int i = 0; i < 3 && why.empty(); ++i) {
            struct stat st{};
            if (stat((gDir + "/" + names[i]).c_str(), &st) != 0 || st.st_mtime != S->mtime[i]) why = std::string(names[i]) + " changed";
        }
    }
    if (!why.empty()) {
        log = "not used (" + why + "), loading from the files";
        star_genome_preload_release();   // this process's references: STAR allocates its own copy next
        return false;
    }
    G1 = S->G1; SA = S->SA; SAi = S->SAi;
    char b[160];
    snprintf(b, sizeof b, "taken from singlet's preload (%.2f s, shared copy-on-write with the processes forked after it)", S->seconds);
    log = b;
    delete S;                 // this process owns the arrays now (Genome::freeMemory deletes them)
    g_genomePreload = nullptr;
    return true;
}
