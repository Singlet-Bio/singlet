// bam_sort_budget.h — memory ceiling for STAR's coordinate BAM sort.
//
// STAR sorts one genomic bin at a time and stops with "not enough memory for BAM sorting" when the largest bin is bigger than
// --limitBAMsortRAM. The bin sizes depend on the data, not on the read count: bins are cut from the first 200k mapped reads and a few
// loci hold a large share of all reads (Kang 2018 stimulated PBMC, 172M reads: largest bin 50.6 GB against the 32 GB the
// disk-sort tier asked for; Kang b1_a, 187M reads: 21.7 GB). The limit is therefore raised at the point of failure, to what the
// bin needs, when the node can hold it. Pure functions so the decision is unit-tested with a tiny budget.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>

namespace singlet { namespace star_sort {

// Memory this process family may use, in bytes: the smallest of SLURM_MEM_PER_NODE, the cgroup limit and MemTotal.
// SINGLET_MEM_BUDGET_BYTES overrides it (tests, and sites with a limit none of the three show). 0 = unknown.
inline uint64_t memory_budget_bytes() {
    if (const char* e = std::getenv("SINGLET_MEM_BUDGET_BYTES"); e && *e) return std::strtoull(e, nullptr, 10);
    uint64_t b = 0;
    auto take = [&](uint64_t v) { if (v > 0 && (b == 0 || v < b)) b = v; };
    if (const char* s = std::getenv("SLURM_MEM_PER_NODE"); s && *s) take(std::strtoull(s, nullptr, 10) * 1024ULL * 1024ULL);
    for (const char* p : {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        std::ifstream f(p); std::string v;
        if (f >> v && !v.empty() && v[0] >= '0' && v[0] <= '9') { uint64_t x = std::strtoull(v.c_str(), nullptr, 10); if (x < (1ULL << 50)) take(x); }
    }
    std::ifstream mi("/proc/meminfo"); std::string k; uint64_t kb;
    if (mi >> k >> kb && k == "MemTotal:") take(kb * 1024ULL);
    return b;
}

// Resident set size of this process (bytes), 0 if unreadable.
inline uint64_t resident_bytes() {
    std::ifstream f("/proc/self/statm"); uint64_t sz = 0, rs = 0;
    if (f >> sz >> rs) return rs * 4096ULL;
    return 0;
}

// Largest limitBAMsortRAM that still leaves a margin (10% of the budget, at least 2 GB) after the memory already in use.
// 0 when the budget is unknown (the caller then does not raise the limit).
inline uint64_t sort_ram_ceiling(uint64_t budget, uint64_t in_use) {
    if (budget == 0) return 0;
    const uint64_t margin = std::max<uint64_t>(budget / 10, 2ULL << 30);
    const uint64_t used = in_use + margin;
    return budget > used ? budget - used : 0;
}

// Decision for a largest-bin size `max_bin` against the configured limit: the limit to use, or 0 = does not fit (fail with the STAR message).
// A bin that fits the configured limit is left alone, so a run that succeeded before is unchanged.
inline uint64_t sort_ram_for_bin(uint64_t max_bin, uint64_t limit, uint64_t budget, uint64_t in_use) {
    if (max_bin <= limit) return limit;
    const uint64_t want = max_bin + (1ULL << 30);          // the bin plus 1 GB, as STAR's own message asks
    const uint64_t ceil = sort_ram_ceiling(budget, in_use);
    return (ceil >= want) ? want : 0;
}

}}  // namespace singlet::star_sort
