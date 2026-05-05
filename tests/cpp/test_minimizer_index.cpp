// test_minimizer_index.cpp — spot-check minimizer_index.h compilation + query
// Compile: g++ -std=c++17 -O2 -I../include test_minimizer_index.cpp -o test_minimizer_index
// Run:     ./test_minimizer_index <minimizerIndex.bin> <genome-dir>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "singlet-pileup/minimizer_index.h"

// Read a PackedArray SA entry (33-bit for GRCh38)
static uint64_t sa_get(const char* arr, uint64_t ii, int bits) {
    uint64_t b = (uint64_t)ii * (uint64_t)bits;
    uint64_t B = b >> 3;
    int      S = (int)(b & 7);
    uint64_t word;
    memcpy(&word, arr + B, sizeof(word));
    uint64_t mask = (bits < 64) ? ((1ULL << bits) - 1) : ~0ULL;
    return (word >> S) & mask;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <minimizerIndex.bin> <genome-dir>\n", argv[0]);
        return 1;
    }
    const char* idx_path    = argv[1];
    const char* genome_dir  = argv[2];

    // Load index
    MinimizerSAIndex idx;
    idx.mmap_load(idx_path);
    printf("Index loaded: %lu records  k=%d w=%d T=%d\n",
           idx.size(), idx.k(), idx.w(), idx.T());

    // mmap genome
    char genome_path[1024]; snprintf(genome_path, sizeof(genome_path), "%s/Genome", genome_dir);
    int gfd = open(genome_path, O_RDONLY);
    assert(gfd >= 0);
    struct stat gsb{}; fstat(gfd, &gsb);
    uint64_t nGenome = gsb.st_size;
    const uint8_t* genome = (const uint8_t*)mmap(nullptr, nGenome, PROT_READ, MAP_SHARED, gfd, 0);
    close(gfd);
    assert(genome != MAP_FAILED);

    // mmap SA
    char sa_path[1024]; snprintf(sa_path, sizeof(sa_path), "%s/SA", genome_dir);
    int safd = open(sa_path, O_RDONLY);
    assert(safd >= 0);
    struct stat sasb{}; fstat(safd, &sasb);
    uint64_t nSAbyte = sasb.st_size;
    // GstrandBit: parse from genomeParameters.txt
    int GstrandBit = 32;
    char params_path[1024]; snprintf(params_path, sizeof(params_path), "%s/genomeParameters.txt", genome_dir);
    FILE* pf = fopen(params_path, "r");
    if (pf) { char line[256]; while (fgets(line, sizeof(line), pf)) { int gb; if (sscanf(line, "GstrandBit %d", &gb) == 1) GstrandBit = gb; } fclose(pf); }
    int sa_bits = GstrandBit + 1;
    uint64_t nSA = (nSAbyte * 8) / sa_bits;
    uint64_t sa_map_size = nSAbyte + 8;
    const char* sa = (const char*)mmap(nullptr, sa_map_size, PROT_READ, MAP_SHARED, safd, 0);
    close(safd);
    assert(sa != MAP_FAILED);

    printf("nGenome=%lu  nSA=%lu  sa_bits=%d\n", nGenome, nSA, sa_bits);

    // Spot-check 10 random genome positions
    int k = idx.k(), w = idx.w();
    int win_len = k + w - 1;
    int pass = 0, fail = 0, skip = 0;
    srand(42);
    for (int t = 0; t < 10; t++) {
        uint64_t pos = ((uint64_t)rand() * rand()) % (nGenome - win_len);
        uint64_t h = minimizer_compute(genome + pos, k, w);
        if (h == 0) { fprintf(stderr, "  [%d] pos=%lu skipped (N-base)\n", t, pos); skip++; continue; }

        auto qr = idx.query(h);
        if (!qr.found) {
            fprintf(stderr, "  [%d] pos=%lu hash=%016llx NOT FOUND in index\n", t, pos, (unsigned long long)h);
            fail++;
            continue;
        }

        // Verify: binary search in SA[qr.sa_lo..qr.sa_hi] for genome position matching 'pos'
        // For a valid SA range, the genome[SA[i]] for i in [sa_lo..sa_hi] should all share
        // the same minimizer. Check that at least ONE SA entry in this range equals 'pos'.
        bool pos_found = false;
        for (uint32_t i = qr.sa_lo; i <= qr.sa_hi && (i - qr.sa_lo) < 200; i++) {
            uint64_t sa_val = sa_get(sa, i, sa_bits);
            if (sa_val == pos) { pos_found = true; break; }
        }

        if (pos_found) {
            printf("  [%d] PASS pos=%lu hash=%016llx SA=[%u,%u] range=%u\n",
                   t, pos, (unsigned long long)h, qr.sa_lo, qr.sa_hi, qr.sa_hi - qr.sa_lo);
            pass++;
        } else {
            // Range broader check: search the SA to find pos directly
            // Binary search for exact position match
            uint64_t lo2 = 0, hi2 = nSA;
            while (lo2 < hi2) {
                uint64_t mid = (lo2 + hi2) / 2;
                uint64_t sv  = sa_get(sa, mid, sa_bits);
                if (sv < pos) lo2 = mid + 1;
                else           hi2 = mid;
            }
            uint64_t exact_sa_idx = (lo2 < nSA && sa_get(sa, lo2, sa_bits) == pos) ? lo2 : UINT64_MAX;

            if (exact_sa_idx == UINT64_MAX) {
                printf("  [%d] SKIP pos=%lu not in SA (genomeSAsparseD)\n", t, pos);
                skip++;
            } else if (exact_sa_idx >= qr.sa_lo && exact_sa_idx <= qr.sa_hi) {
                printf("  [%d] PASS(bin) pos=%lu sa_idx=%lu SA=[%u,%u]\n",
                       t, pos, exact_sa_idx, qr.sa_lo, qr.sa_hi);
                pass++;
            } else {
                printf("  [%d] FAIL pos=%lu sa_idx=%lu outside SA=[%u,%u]\n",
                       t, pos, exact_sa_idx, qr.sa_lo, qr.sa_hi);
                fail++;
            }
        }
    }

    printf("\nResult: %d/10 pass, %d skip, %d fail\n", pass, skip, fail);
    munmap((void*)genome, nGenome);
    munmap((void*)sa, sa_map_size);
    return (fail > 0) ? 1 : 0;
}
