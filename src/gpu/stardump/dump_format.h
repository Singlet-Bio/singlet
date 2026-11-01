// SPDX-License-Identifier: MIT
// Binary layout of the STAR seed-search dump (W11). One file per STAR mapping thread: <prefix>.<tid>.bin.
//
//   FileHeader
//   then a stream of records, each starting with a one-byte tag:
//     ReadRec  'R'  once per read, before its first seed call:
//                   ReadRec, Nsplit x uint16[3] {start, length, fragment} (STAR's splitR after qualitySplit),
//                   Lread bytes Read1[0] (numeric read, 0..3 = ACGT, >3 = N), Lread bytes Read1[1] (STAR's complement)
//     SeedRec  'S'  one per maxMappableLength2strands call and iDist, in STAR's call order for that read
//
// Reads that never reach the seed search (no good pieces) have no record. All fields little-endian, packed.
#pragma once
#include <cstdint>

namespace stardump {

enum : uint8_t { BR_SAI_ONLY = 0, BR_UNIQUE_EXTEND = 1, BR_SA_SEARCH = 2 };

#pragma pack(push, 1)
struct FileHeader {
    char magic[8] = {'S', 'T', 'A', 'R', 'D', 'M', 'P', '1'};
    uint32_t version = 1;
    uint32_t seedRecBytes = 44;
};

struct ReadRec {
    uint8_t tag = 'R';
    uint64_t iReadAll;     // STAR's global read index (input order)
    uint32_t count;        // multiplicity of this distinct sequence in the sample (from the read name)
    uint16_t Lread;
    uint8_t Nsplit;
};

struct SeedRec {
    uint8_t tag = 'S';
    uint8_t iDir;          // 0 = forward search from pieceStart, 1 = reverse search leftward from pieceStart
    uint8_t iFrag;
    uint8_t branch;        // BR_*: which branch of maxMappableLength2strands answered
    uint8_t iDist;
    uint16_t pieceStart;   // Shift
    uint16_t pieceLength;  // seedLength
    uint8_t Lind;          // longest SAindex prefix present
    uint16_t maxL;         // result: maximal mappable length
    uint64_t Nrep;         // result: number of loci
    uint64_t ind0, ind1;   // result: SA interval [ind0, ind1]
    uint64_t sa0;          // SA[ind0] (strand bit << GstrandBit | position), ~0 if Nrep == 0
};
#pragma pack(pop)

static_assert(sizeof(ReadRec) == 16, "ReadRec layout");
static_assert(sizeof(SeedRec) == 44, "SeedRec layout");

void closeAll();   // flush and close every thread's dump file (call after star_main_impl returns)

} // namespace stardump
