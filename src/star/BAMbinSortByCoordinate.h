#ifndef CODE_BAMbinSortByCoordinate
#define CODE_BAMbinSortByCoordinate
#include "IncludeDefine.h"
#include "Parameters.h"
#include "Genome.h"
#include "Solo.h"

#include SAMTOOLS_BGZF_H

#include <vector>

void BAMbinSortByCoordinate(uint32 iBin, uint binN, uint binS, uint nThreads, string dirBAMsort, Parameters &P, Genome &mapGen, Solo &solo);

// Sort one bin and compress it into BGZF blocks in memory (no header, no EOF block), ready to be appended to the
// coordinate-sorted BAM with bgzf_raw_write(). Same records, same order as BAMbinSortByCoordinate().
void BAMbinSortByCoordinateToBuffer(uint32 iBin, uint binN, uint binS, uint nThreads, string dirBAMsort, Parameters &P,
                                    Genome &mapGen, Solo &solo, std::vector<char> &out);

#endif