// SPDX-License-Identifier: MIT
// tests/cpp/test_gpu_solo_twopass.cpp
//
// Solo's counting in two passes over read chunks (e9::soloRunTwoPass: pass 1 per chunk, pass 2 per key bucket) against the one-shot device path
// (e9::soloRun(device)) on synthetic reads, field by field. The fixture has what the two passes must get right: reads whose CR matches a whitelist
// entry exactly, with one mismatch, with one N (several candidates when whitelist entries differ only at that base), with two Ns and with no
// match; UMIs with an N or a homopolymer and a few UMIs with Hamming-1 neighbors (the 1MM_All collapse); reads without a record whose annotation
// comes from the host (sparse); reads with several matches whose posterior reads per-read CB qualities (ascending index, batched). Run with
// several chunk sizes and bucket counts, with the per-record annotations in host memory, with and without per-read qualities.
// Prints SKIP and returns 77 (ctest: skipped) in a CPU-only build or without a usable CUDA device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "test_harness.h"
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/engine_gpu.h"
#include "solo_gpu.h"
#endif

#ifdef SINGLET_HAVE_ENGINE_GPU
namespace {
using namespace e9;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; uint64_t x = s; x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; return x; }
    uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
    double unit() { return (double)(next() >> 11) / 9007199254740992.0; }
};

constexpr u32 CBL = 16, UMIL = 12;

// bases (0..3, 4 = N), first base lowest, 3 bits each (HandoffReadEntry packing)
u64 pack3(const std::vector<int>& b) { u64 v = 0; for (size_t j = 0; j < b.size(); j++) v |= (u64)b[j] << (3 * j); return v; }
u64 wlToCr(u64 w) { std::vector<int> b(CBL); for (u32 j = 0; j < CBL; j++) b[j] = (int)((w >> (2 * (CBL - 1 - j))) & 3); return pack3(b); }

struct Fixture {
    SoloParams S;
    SoloModel M;
    std::vector<RecAnn> recAnn;
    std::vector<u64> cr3, ur3;
    std::vector<u32> rec;
    std::vector<u32> hostIdx;
    std::vector<RecAnn> hostAnn;
};

RecAnn randomAnn(Rng& r, u32 nGe) {
    RecAnn a{0xFFFFFFFFu, 0, 0, 0};
    const double u = r.unit();
    a.nTr = u < 0.05 ? 0 : u < 0.85 ? 1 : 3;
    if (a.nTr == 0) return a;
    a.bits = (uint8_t)(r.below(4));
    const double g = r.unit();
    if (g < 0.1) { a.ng = 0; }
    else if (g < 0.2) { a.ng = 2; }
    else { a.ng = 1; a.gene = r.below(nGe); }
    return a;
}

Fixture makeFixture(u64 N, u32 W0, uint64_t seed) {
    Fixture f;
    Rng r(seed);
    // whitelist (2 bits per base): random entries, a third of them with a twin that differs in one base
    std::vector<u64> wl;
    std::vector<std::pair<u64, u32>> twinPos;   // (entry, differing position)
    for (u32 i = 0; i < W0; i++) {
        const u64 w = r.next() & 0xffffffffULL;
        wl.push_back(w);
        if (r.below(3) == 0) {
            const u32 pos = r.below(CBL);
            const u64 t = w ^ ((u64)(1 + r.below(3)) << (2 * pos));
            wl.push_back(t);
            twinPos.push_back({w, pos});
        }
    }
    std::sort(wl.begin(), wl.end());
    wl.erase(std::unique(wl.begin(), wl.end()), wl.end());
    f.M.wl = wl;
    f.M.pqTab.resize(256);
    for (int c = 0; c < 256; c++) { int qin = std::min((int)(signed char)c - 33, 40); f.M.pqTab[c] = std::pow(10.0, -qin / 10.0); }
    f.S.cbL = CBL; f.S.umiL = UMIL; f.S.strand = 0; f.S.nGe = 200; f.S.cbMinP = 0.975f; f.S.pq = std::pow(10.0, -3.0);
    f.S.umiMaskLow = (1u << UMIL) - 1;
    f.S.homo[0] = 0; f.S.homo[1] = 0x555555; f.S.homo[2] = 0xAAAAAA; f.S.homo[3] = 0xFFFFFF;
    const u32 W = (u32)wl.size();
    // records
    const u32 nRec = 400;
    for (u32 i = 0; i < nRec; i++) f.recAnn.push_back(randomAnn(r, f.S.nGe));
    // a small pool of UMIs, with Hamming-1 neighbors
    std::vector<u64> umiPool;
    for (int i = 0; i < 24; i++) {
        std::vector<int> b(UMIL);
        for (u32 j = 0; j < UMIL; j++) b[j] = (int)r.below(4);
        umiPool.push_back(pack3(b));
        if (i % 3 == 0) { b[r.below(UMIL)] = (int)r.below(4); umiPool.push_back(pack3(b)); }
    }
    // hot cells so that some CBs collect many reads
    const u32 hot = std::min<u32>(W, 40);
    for (u64 i = 0; i < N; i++) {
        u64 w; u32 twinAt = 99;
        if (r.below(100) < 50) w = wl[r.below(hot)];
        else if (!twinPos.empty() && r.below(100) < 20) { const auto& t = twinPos[r.below((u32)twinPos.size())]; w = t.first; twinAt = t.second; }
        else w = wl[r.below(W)];
        u64 cr = wlToCr(w);
        const u32 kind = r.below(100);
        if (kind < 60) {}                                                                         // exact
        else if (kind < 75) {                                                                     // one mismatch (often at a twin's differing base)
            const u32 pos = (twinAt != 99 && r.below(100) < 70) ? twinAt : r.below(CBL);
            const u64 old = (cr >> (3 * pos)) & 7;
            cr = (cr & ~((u64)7 << (3 * pos))) | ((u64)((old + 1 + r.below(3)) & 3) << (3 * pos));
        } else if (kind < 90) {                                                                   // one N (several candidates when a twin differs there)
            const u32 pos = (twinAt != 99 && r.below(100) < 80) ? twinAt : r.below(CBL);
            cr = (cr & ~((u64)7 << (3 * pos))) | ((u64)4 << (3 * pos));
        } else if (kind < 93) {                                                                   // two Ns
            cr = (cr & ~((u64)63 << 0)) | ((u64)4 << 0) | ((u64)4 << 3);
        } else {                                                                                  // no match
            std::vector<int> b(CBL);
            for (u32 j = 0; j < CBL; j++) b[j] = (int)r.below(4);
            cr = pack3(b);
        }
        std::vector<int> ub(UMIL);
        u64 ur;
        const u32 uk = r.below(100);
        if (uk < 2) { for (u32 j = 0; j < UMIL; j++) ub[j] = 1; ur = pack3(ub); }                // homopolymer
        else if (uk < 4) { for (u32 j = 0; j < UMIL; j++) ub[j] = (int)r.below(4); ub[r.below(UMIL)] = 4; ur = pack3(ub); }   // N
        else if (uk < 70) ur = umiPool[r.below((u32)umiPool.size())];
        else { for (u32 j = 0; j < UMIL; j++) ub[j] = (int)r.below(4); ur = pack3(ub); }
        f.cr3.push_back(cr); f.ur3.push_back(ur);
        if (r.below(100) < 8) {   // the host mapped it: no record, the annotation comes sparse
            f.rec.push_back(0xFFFFFFFFu);
            f.hostIdx.push_back((u32)i);
            f.hostAnn.push_back(randomAnn(r, f.S.nGe));
        } else f.rec.push_back(r.below(nRec));
    }
    return f;
}

bool same(const SoloResult& a, const SoloResult& b, std::string& why) {
    auto eq = [&](const char* n, bool ok) { if (!ok && why.empty()) why = n; return ok; };
    bool ok = true;
    for (u32 i = 0; i < B_N; i++) ok &= eq("bstats", a.bstats[i] == b.bstats[i]);
    for (u32 i = 0; i < S_N; i++) ok &= eq("fstats", a.fstats[i] == b.fstats[i]);
    for (u32 i = 0; i < F_N; i++) ok &= eq("noCB", a.noCB[i] == b.noCB[i]);
    ok &= eq("cbIdx", a.cbIdx == b.cbIdx);
    ok &= eq("cbVal", a.cbVal == b.cbVal);
    ok &= eq("segCb", a.segCb == b.segCb);
    ok &= eq("segGene", a.segGene == b.segGene);
    ok &= eq("segCount", a.segCount == b.segCount);
    ok &= eq("nKeys", a.nKeys == b.nKeys);
    ok &= eq("nTuples", a.nTuples == b.nTuples);
    ok &= eq("nSeg", a.nSeg == b.nSeg);
    ok &= eq("nMulti", a.nMulti == b.nMulti);
    ok &= eq("nMultiAccepted", a.nMultiAccepted == b.nMultiAccepted);
    ok &= eq("perReadQual", a.perReadQual == b.perReadQual);
    return ok;
}

// the qualities the fake barcode FASTQ gives read idx
void fakeQual(u64 idx, u32 cbL, uint8_t* o) {
    Rng r(idx * 7919 + 13);
    for (u32 j = 0; j < cbL; j++) o[j] = (uint8_t)(33 + r.below(41));
}

int run() {
    const Fixture f = makeFixture(60000, 3000, 12345);
    const u64 N = f.cr3.size();
    SoloRecs R; R.nRec = f.recAnn.size(); R.dev = false; R.recAnn = f.recAnn.data();
    auto mkReads = [&](bool perRead, u64* nCalls, bool* ascending) {
        SoloReads Rd; Rd.n = N; Rd.dev = false; Rd.cr3 = f.cr3.data(); Rd.ur3 = f.ur3.data(); Rd.rec = f.rec.data();
        Rd.nHost = f.hostIdx.size(); Rd.hostIdx = f.hostIdx.data(); Rd.hostAnnS = f.hostAnn.data();
        if (perRead) {
            const u32 cbL = f.S.cbL;
            Rd.cbQual = [&f, cbL, nCalls, ascending](const u64* idx, const u64* cr3, u64 n, uint8_t* out, std::string& why) {
                (void)why;
                if (nCalls) (*nCalls)++;
                for (u64 k = 0; k < n; k++) {
                    if (k > 0 && idx[k] <= idx[k - 1] && ascending) *ascending = false;
                    if (cr3[k] != (f.cr3[idx[k]] & ((1ULL << (3 * cbL)) - 1))) { why = "wrong CR"; return false; }
                    fakeQual(idx[k], cbL, out + k * cbL);
                }
                return true;
            };
        }
        return Rd;
    };
    int nFail = 0;
    for (int perRead = 0; perRead < 2; perRead++) {
        std::string log1;
        SoloResult ref;
        u64 calls1 = 0; bool asc1 = true;
        const bool ok1 = soloRun(true, f.S, f.M, R, mkReads(perRead != 0, &calls1, &asc1), ref, false, log1);
        CHECK(ok1, std::string("one-shot device path ran (") + (perRead ? "per-read qualities" : "one quality factor") + ")");
        if (!ok1) { std::cout << log1; return 1; }
        CHECK(ref.nKeys > 1000 && ref.nMulti > 50 && ref.nMultiAccepted > 0, "the fixture exercises counted reads and the posterior");
        std::cout << "  fixture: " << N << " reads, " << ref.nKeys << " counted, " << ref.nMulti << " with several matches (" << ref.nMultiAccepted << " given a CB), " << ref.nTuples << " tuples, " << ref.nSeg << " (CB,gene)
";
        CHECK(ref.nSeg > 100 && ref.nTuples > ref.nSeg, "the fixture has several UMIs per (CB, gene)");
        struct Cfg { const char* name; u64 chunk, bucketKeys, bucketTarget; bool recHost; };
        const Cfg cfgs[] = {
            {"one chunk, one bucket", 0, 0, 0, false},
            {"chunks of 5000, 4 buckets", 5000, 0, 4, false},
            {"chunks of 777, 7 buckets", 777, 0, 7, false},
            {"chunks of 12345, buckets of at most 3000 keys", 12345, 3000, 0, false},
            {"chunks of 4096, records' annotations in host memory", 4096, 0, 5, true},
        };
        for (const Cfg& c : cfgs) {
            SoloTwoPass opt; opt.chunkReads = c.chunk; opt.bucketKeys = c.bucketKeys; opt.bucketTarget = c.bucketTarget; opt.recAnnHost = c.recHost;
            std::string log2;
            SoloResult got;
            u64 calls2 = 0; bool asc2 = true;
            const bool ok2 = soloRunTwoPass(f.S, f.M, R, mkReads(perRead != 0, &calls2, &asc2), got, log2, opt);
            CHECK(ok2, std::string("two-pass path ran: ") + c.name);
            if (!ok2) { std::cout << log2; nFail++; continue; }
            std::string why;
            const bool eq = same(ref, got, why);
            CHECK(eq, std::string("identical to the one-shot path: ") + c.name + (eq ? "" : " (differs in " + why + ")"));
            if (c.chunk) CHECK(got.nChunks == (N + c.chunk - 1) / c.chunk, std::string("chunk count: ") + c.name);
            if (c.bucketTarget >= 4) CHECK(got.nBuckets >= 2, std::string("several buckets: ") + c.name);
            if (c.bucketKeys == 3000) CHECK(got.nBuckets >= ref.nKeys / 3000, std::string("bucket size bound: ") + c.name);
            if (perRead) CHECK(calls2 >= 1 && asc2, std::string("the quality callback was called, in ascending order: ") + c.name);
            if (!eq) std::cout << log2;
        }
        // a bucket smaller than one CB's keys fails loudly (never truncates)
        SoloTwoPass opt; opt.chunkReads = 5000; opt.bucketKeys = 1;
        std::string log3; SoloResult bad;
        const bool ok3 = soloRunTwoPass(f.S, f.M, R, mkReads(perRead != 0, nullptr, nullptr), bad, log3, opt);
        CHECK(!ok3 && log3.find("more than a bucket") != std::string::npos, "a bucket smaller than a CB's keys fails with a message");
    }
    return n_fail == 0 && nFail == 0 ? 0 : 1;
}
}  // namespace
#endif

int main() {
#ifndef SINGLET_HAVE_ENGINE_GPU
    std::cout << "SKIP: CPU-only build (no GPU engine)\n";
    return 77;
#else
    std::string why;
    if (singlet_engine_gpu::deviceCount() <= 0 || !singlet_engine_gpu::available(&why)) {
        std::cout << "SKIP: no usable CUDA device" << (why.empty() ? "" : " (" + why + ")") << "\n";
        return 77;
    }
    const int rc = run();
    std::cout << (rc == 0 ? "ALL PASSED" : "FAILED") << " (" << n_pass << " checks passed, " << n_fail << " failed)\n";
    return rc;
#endif
}
