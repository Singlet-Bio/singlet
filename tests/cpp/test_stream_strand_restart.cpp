// SPDX-License-Identifier: MIT
// tests/cpp/test_stream_strand_restart.cpp
//
// Stream path (order-free-rules R5) counts the whole pass in the protocol orientation; a library far from it (Visium FFPE probes,
// reverse-stranded chemistries) used to abort the run at the end of the pass with exit 8 ("rerun with --sorted-bam"). The pass now
// stops after the first million reads when the evidence is clear and singlet restarts the same command on the sorted path.
// This checks the decision: enough informative reads, and a clear majority of them antisense to the orientation being counted.

#include <cassert>
#include <cstdio>

#include "singlet/pileup/pileup_engine.h"

using singlet::PileupStats;

static PileupStats stats_with(uint64_t sense, uint64_t antisense) {
    PileupStats s;
    s.strand_sense = sense;
    s.strand_antisense = antisense;
    return s;
}

int main() {
    // Visium FFPE: 99% of the informative reads are antisense to the forward protocol orientation -> stop
    assert(singlet::strand_early_abort_due(stats_with(2000, 198000), /*reverse=*/false));
    // the same library counted in the right (reverse) orientation: the sense counts are the "wrong" ones -> keep going
    assert(!singlet::strand_early_abort_due(stats_with(2000, 198000), /*reverse=*/true));
    // and a reverse-counted pass over a forward library stops
    assert(singlet::strand_early_abort_due(stats_with(198000, 2000), /*reverse=*/true));
    // an ordinary 3' library (a few percent antisense) never stops
    assert(!singlet::strand_early_abort_due(stats_with(190000, 10000), false));
    // not enough informative reads yet: keep going whatever the share
    assert(!singlet::strand_early_abort_due(stats_with(0, 199999), false));
    assert(singlet::strand_early_abort_due(stats_with(0, 200000), false));
    // a mixed library (60-90% antisense) is left to the end-of-pass check, which restarts on the aggregate
    assert(!singlet::strand_early_abort_due(stats_with(80000, 120000), false));
    assert(!singlet::strand_early_abort_due(stats_with(100000, 900000), false));   // 90.0% exactly: not above the cut
    std::printf("test_stream_strand_restart: PASS\n");
    return 0;
}
