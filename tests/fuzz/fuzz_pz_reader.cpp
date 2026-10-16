// SPDX-License-Identifier: MIT
// libFuzzer target for the .1pz reader.
//
//   clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -Iinclude \
//       tests/fuzz/fuzz_pz_reader.cpp -lzstd -o fuzz_pz_reader
//   ./fuzz_pz_reader corpus/ -max_len=65536 -timeout=10
//
// The whole-file CRC32 is repaired before decoding so the fuzzer reaches the
// structural checks instead of stopping at the checksum.

#include "singlet/pileup/pz_reader.h"
#include "singlet/pileup/pz_writer.h"

#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    namespace pz = singlet::pz;
    std::vector<uint8_t> buf(data, data + size);
    if (buf.size() >= sizeof(pz::PZHeader) + sizeof(pz::PZFooter)) {
        uint32_t crc = pz::CRC32::compute(buf.data(), buf.size() - sizeof(pz::PZFooter));
        std::memcpy(buf.data() + buf.size() - sizeof(pz::PZFooter), &crc, 4);
        uint32_t magic = pz::TP1_MAGIC;
        std::memcpy(buf.data(), &magic, 4);
        std::memcpy(buf.data() + buf.size() - 4, &magic, 4);
    }
    pz::ReadLimits lim;
    lim.max_rows = 1u << 16;
    lim.max_cols = 1u << 16;
    lim.max_nnz  = 1u << 22;
    try {
        (void)pz::read_1pz_buffer(buf.data(), buf.size(), lim);
    } catch (const pz::PZReadError&) {
    }
    return 0;
}
