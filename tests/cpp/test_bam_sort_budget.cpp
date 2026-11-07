// STAR BAM-sort memory ceiling (include/singlet/star/bam_sort_budget.h): the limit is raised for a bin that does not fit
// the configured limit only when the node's budget holds it, and never otherwise.

#include <cstdlib>
#include <iostream>

#include "star/bam_sort_budget.h"

using namespace singlet::star_sort;

static int n_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::cout << "FAIL line " << __LINE__ << ": " #cond "\n"; ++n_fail; } } while (0)

int main() {
    const uint64_t GB = 1ULL << 30;
    // Kang stim on a 200 GB node: bin 50.6e9 bytes against the 32 GB disk-tier limit, 25 GB in use -> raised to the bin plus 1 GB.
    const uint64_t bin = 50621706147ULL;
    CHECK(sort_ram_for_bin(bin, 32 * GB, 200 * GB, 25 * GB) == bin + GB);
    // A bin that fits the configured limit leaves it alone (a run that worked before is unchanged).
    CHECK(sort_ram_for_bin(21 * GB, 32 * GB, 200 * GB, 25 * GB) == 32 * GB);
    CHECK(sort_ram_for_bin(21 * GB, 32 * GB, 8 * GB, 7 * GB) == 32 * GB);
    // Tiny budget: the raise does not fit, 0 = fail with STAR's message instead of an OOM kill.
    CHECK(sort_ram_for_bin(bin, 32 * GB, 64 * GB, 25 * GB) == 0);
    CHECK(sort_ram_for_bin(bin, 32 * GB, 4 * GB, 1 * GB) == 0);
    CHECK(sort_ram_for_bin(bin, 32 * GB, 0, 0) == 0);                 // unknown budget: no raise
    // Ceiling keeps 10% of the budget (at least 2 GB) free after what is in use.
    CHECK(sort_ram_ceiling(200 * GB, 25 * GB) == 155 * GB);
    CHECK(sort_ram_ceiling(10 * GB, 1 * GB) == 7 * GB);               // margin = max(1 GB, 2 GB)
    CHECK(sort_ram_ceiling(10 * GB, 9 * GB) == 0);
    // The environment override feeds memory_budget_bytes.
    putenv((char*)"SINGLET_MEM_BUDGET_BYTES=1000000");
    CHECK(memory_budget_bytes() == 1000000ULL);
    CHECK(sort_ram_for_bin(bin, 32 * GB, memory_budget_bytes(), 0) == 0);
    if (n_fail == 0) std::cout << "all passed\n";
    return n_fail ? 1 : 0;
}
