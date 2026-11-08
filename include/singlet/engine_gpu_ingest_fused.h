// SPDX-License-Identifier: MIT
#pragma once
// The fused .1fq pass (SeqCacheGpuIngest.cpp decodeScan): the pipeline's head scan and the packed-read decode (engine_gpu_ingest.h phase A)
// from one read and one decompression of each block. It fills the head scan's per-block results and counters exactly as RunSet::scan_head
// does, and the packed reads exactly as decodeReads does after a head scan; startParent then finds the reads decoded. The arrays are reserved
// before any block is decoded, from the block headers and the header's cDNA length (or the payloads' sizes in their ZSTD frame headers). dropFused releases the reads when the pipeline
// does not use the ingest after all (a 5' prefix to clip).
#include <cstdint>
#include <string>
#include <vector>
#include "singlet/engine_gpu_ingest.h"
#include "singlet/fq/read_source.h"

namespace singlet_engine_gpu {
namespace ingest {
bool decodeScan(int threads, const singlet::fq::RunSet::HeadScanOptions& so, std::vector<singlet::fq::RunSet::BlockHead>& heads, uint32_t r2Fixed,
                std::string& why);
void dropFused();
}  // namespace ingest
}  // namespace singlet_engine_gpu
