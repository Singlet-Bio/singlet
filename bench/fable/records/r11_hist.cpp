// E6: STAR's exact CB histogram (cbReadCountExact) of a .1fq, as singlet's decode pass builds it for the R11 correction of the stream path
// (src/pipeline/singlet.cpp, "[1fq-decode] R11 exact CB histogram": reads given to STAR whose CB is an exact whitelist hit and whose UMI has
// no N and is not a homopolymer, weighted by duplicate counts; barcode-dictionary reads counted by slot and folded by the slot's barcode).
// Output for records_test --r11-hist: "E6R11H01", u64 whitelist size, u64 total, u64 n pairs, then n x (u64 whitelist index, u64 count).
// Only barcode counts are written (no read sequence).
#include "singlet/fq/reader.h"
#include "singlet/pileup/cb_correct.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: r11_hist <in.1fq> <whitelist> <out.bin> [cbS cbL umiS umiL]\n"); return 2; }
    const int cbS = argc > 4 ? std::atoi(argv[4]) : 1, cbl = argc > 5 ? std::atoi(argv[5]) : 16, umiS = argc > 6 ? std::atoi(argv[6]) : 17,
              umil = argc > 7 ? std::atoi(argv[7]) : 12;
    const int cb0 = cbS - 1, umi0 = umiS - 1;
    singlet::CbCorrector cc;
    if (!cc.load_whitelist(argv[2]) || cc.length() != cbl) { std::fprintf(stderr, "whitelist %s unusable for %d-base barcodes\n", argv[2], cbl); return 2; }
    singlet::fq::Reader r;
    r.open(argv[1]);
    const bool hasDict = r.has_bc_dict();
    const uint32_t dictN = r.bc_dict_size();
    const uint16_t dlen = r.bc_length();
    std::vector<uint64_t> dictSum(hasDict ? dictN : 0, 0);
    std::unordered_map<int64_t, uint64_t> raw;
    static constexpr char kNt[5] = {'A', 'C', 'G', 'T', 'N'};
    singlet::fq::DecodedBlock blk;
    uint64_t nStored = 0, nGiven = 0;
    while (r.read_block(blk)) {
        const bool hasDups = !blk.dup_counts.empty();
        for (uint32_t i = 0; i < blk.n_reads; ++i) {
            nStored++;
            const uint64_t repeat = hasDups ? blk.dup_counts[i] : 1;
            const uint16_t r1len = blk.r1_len(i);
            if (r1len == 0 || blk.r2_len(i) == 0) continue;   // not given to STAR (format_block)
            nGiven += repeat;
            if (!(r1len >= cb0 + cbl && r1len >= umi0 + umil)) continue;
            const uint8_t* r1 = blk.r1_seq(i);
            const uint8_t* u = r1 + umi0;
            bool umiOk = true, homo = true;
            for (int j = 0; j < umil; ++j) {
                if (u[j] > 3) { umiOk = false; break; }
                if (u[j] != u[0]) homo = false;
            }
            if (!umiOk || homo) continue;
            const uint32_t idx = (hasDict && i < blk.bc_indices.size()) ? blk.bc_indices[i] : UINT32_MAX;
            if (idx < dictN) { dictSum[idx] += repeat; continue; }
            char bcs[32];
            for (int j = 0; j < cbl; ++j) bcs[j] = kNt[r1[cb0 + j] < 4 ? r1[cb0 + j] : 4];
            const int64_t w = cc.find(bcs);
            if (w >= 0) raw[w] += repeat;
        }
    }
    std::vector<uint64_t> hist(cc.size(), 0);
    uint64_t total = 0;
    if (hasDict && dlen == static_cast<uint16_t>(cbl)) {
        const std::vector<uint8_t>& dflat = r.bc_dict_flat();
        char bcs[32];
        for (uint32_t k = 0; k < dictN; ++k) {
            if (!dictSum[k]) continue;
            for (int j = 0; j < cbl; ++j) { const uint8_t c = dflat[static_cast<size_t>(k) * dlen + j]; bcs[j] = kNt[c < 4 ? c : 4]; }
            const int64_t w = cc.find(bcs);
            if (w >= 0) { hist[w] += dictSum[k]; total += dictSum[k]; }
        }
    }
    for (const auto& kv : raw) { hist[kv.first] += kv.second; total += kv.second; }
    uint64_t nnz = 0;
    for (uint64_t v : hist) nnz += v != 0;
    FILE* f = std::fopen(argv[3], "wb");
    if (!f) { std::perror(argv[3]); return 3; }
    const uint64_t h[3] = {static_cast<uint64_t>(cc.size()), total, nnz};
    bool ok = std::fwrite("E6R11H01", 1, 8, f) == 8 && std::fwrite(h, 8, 3, f) == 3;
    for (uint64_t w = 0; ok && w < hist.size(); ++w)
        if (hist[w]) { const uint64_t p[2] = {w, hist[w]}; ok = std::fwrite(p, 8, 2, f) == 2; }
    ok = (std::fclose(f) == 0) && ok;
    std::printf("r11_hist: %llu stored reads, %llu given to STAR (with duplicates); exact CB histogram %llu reads over a %zu-barcode whitelist, %llu barcodes;"
                " barcode dictionary %s (%u slots); -> %s %s\n", (unsigned long long)nStored, (unsigned long long)nGiven, (unsigned long long)total, cc.size(),
                (unsigned long long)nnz, hasDict ? "yes" : "no", dictN, argv[3], ok ? "written" : "FAILED");
    return ok ? 0 : 3;
}
