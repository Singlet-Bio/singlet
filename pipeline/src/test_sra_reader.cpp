// test_sra_reader.cpp — quick standalone test for SraReader
// Usage: ./test_sra_reader FILE.sra [N_SPOTS]
// Writes first N_SPOTS (default 10) to stdout as R1+R2 FASTQ

#include "sra_reader.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s FILE.sra [N_SPOTS]\n", argv[0]);
        return 1;
    }
    int n = (argc > 2) ? atoi(argv[2]) : 10;

    SraReader reader;
    reader.open(argv[1]);
    fprintf(stderr, "Opened: %lld spots\n", (long long)reader.total_spots());

    // Write to stdout files for R1 and R2
    FILE* r1 = fopen("/dev/stdout", "w");
    FILE* r2 = fopen("/dev/stderr", "w");  // R2 goes to stderr for quick check

    int count = 0;
    while (!reader.done() && count < n) {
        reader.write_next_spot(r1, r2);
        count++;
    }

    fclose(r1);
    fprintf(stderr, "\nWrote %d spots\n", count);
    return 0;
}
