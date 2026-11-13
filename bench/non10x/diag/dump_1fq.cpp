// SPDX-License-Identifier: MIT
// D-152 diagnostic: print the first reads (R1 as stored, R2) and the N fraction of R2 of a .1fq. usage: dump_1fq FILE [N]
#include <iostream>
#include "singlet/fq/lib1fq.h"
using namespace singlet::fq;
int main(int argc, char** argv) {
    Reader rd; rd.open(argv[1]);
    std::cout << "stream_lengths " << rd.header().stream_lengths[0] << " " << rd.header().stream_lengths[1] << " flags " << (int)rd.header().flags << "\n";
    DecodedBlock b; uint64_t reads = 0, bases = 0, nb = 0; int shown = 0, want = argc > 2 ? std::atoi(argv[2]) : 6;
    while (rd.read_block(b)) {
        for (uint32_t i = 0; i < b.n_reads; ++i) {
            const uint8_t* s2 = b.r2_data.data() + b.r2_offsets[i]; const uint16_t L2 = b.r2_lengths[i];
            const uint8_t* s1 = b.r1_data.data() + b.r1_offsets[i]; const uint16_t L1 = b.r1_lengths[i];
            ++reads; bases += L2; for (uint16_t j = 0; j < L2; ++j) nb += (s2[j] > 3);
            if (shown < want && reads % 5001 == 1) {
                std::string a, c; for (uint16_t j = 0; j < L1; ++j) a.push_back("ACGTN"[s1[j] > 4 ? 4 : s1[j]]); for (uint16_t j = 0; j < L2; ++j) c.push_back("ACGTN"[s2[j] > 4 ? 4 : s2[j]]);
                std::cout << "R1(" << L1 << ") " << a << "\nR2(" << L2 << ") " << c << "\n"; ++shown;
            }
        }
    }
    std::cout << "reads " << reads << " mean R2 " << (double)bases / std::max<uint64_t>(1, reads) << " N fraction " << (double)nb / std::max<uint64_t>(1, bases) << "\n";
}
