// The 2-bit read store with an exact sparse N record (engine_gpu.h PackedSeq, PackedSeqWriter): what goes in comes out byte for byte.
// Random blocks of random reads (lengths 0 to 200, every alignment of a read's start within a byte), N runs of every length, reads of all N,
// runs touching read and block edges, empty blocks; every read, every span over block edges, and the whole store are compared with the input.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "singlet/engine_gpu.h"

using singlet_engine_gpu::PackedSeq;
using singlet_engine_gpu::PackedSeqWriter;

static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s\n", msg); ++n_fail; } } while (0)

int main() {
    std::mt19937_64 rng(12345);
    for (int round = 0; round < 40; round++) {
        const int nB = 1 + (int) (rng() % 9);
        std::vector<char> all;                          // the original codes
        std::vector<uint64_t> off{0};                   // read offsets
        PackedSeq ps;
        ps.baseAt.assign(1, 0); ps.byteAt.assign(1, 0); ps.runs.resize(nB);
        std::vector<std::vector<char>> blocks(nB);
        const int style = round % 4;                    // 0 no N, 1 sparse N, 2 dense N runs, 3 mixed with all-N reads
        for (int b = 0; b < nB; b++) {
            const int nr = (b == 1 && round % 5 == 0) ? 0 : (int) (rng() % 60);
            std::vector<char>& blk = blocks[b];
            for (int r = 0; r < nr; r++) {
                const int L = (int) (rng() % 201);
                std::vector<char> rd(L);
                for (int j = 0; j < L; j++) rd[j] = (char) (rng() & 3);
                if (style >= 1 && L > 0) {
                    const int nruns = style == 1 ? (int) (rng() % 2) : (int) (rng() % 5);
                    for (int u = 0; u < nruns; u++) {
                        const int a = (int) (rng() % L), len = 1 + (int) (rng() % (style == 1 ? 3 : 40));
                        for (int j = a; j < L && j < a + len; j++) rd[j] = 4;
                    }
                    if (style == 3 && rng() % 6 == 0) for (int j = 0; j < L; j++) rd[j] = 4;
                    if (style == 3 && rng() % 8 == 0) rd[0] = 4;
                    if (style == 3 && rng() % 8 == 0) rd[L - 1] = 4;
                }
                blk.insert(blk.end(), rd.begin(), rd.end());
                off.push_back(off.back() + L);
            }
            all.insert(all.end(), blk.begin(), blk.end());
            ps.baseAt.push_back(ps.baseAt.back() + blk.size());
            ps.byteAt.push_back(ps.byteAt.back() + (blk.size() + 3) / 4);
        }
        ps.total = ps.baseAt.back();
        ps.w.resize(std::max<uint64_t>(1, ps.byteAt.back()));
        // pack each block read by read, as the ingest does
        uint64_t rdAt = 0;
        for (int b = 0; b < nB; b++) {
            PackedSeqWriter w(ps.w.data() + ps.byteAt[b], ps.baseAt[b], &ps.runs[b]);
            uint64_t pos = ps.baseAt[b];
            while (rdAt + 1 < off.size() && off[rdAt] >= ps.baseAt[b] && off[rdAt + 1] <= ps.baseAt[b + 1] && pos < ps.baseAt[b + 1]) {
                const uint64_t L = off[rdAt + 1] - off[rdAt];
                w.put((const uint8_t*) all.data() + off[rdAt], (uint32_t) L);
                pos += L; rdAt++;
            }
            // reads of length 0 at the block's end belong to it too
            w.finish();
            CHECK(!w.bad, "codes outside 0-4 reported");
        }
        // every read
        std::vector<char> buf(256);
        for (size_t r = 0; r + 1 < off.size(); r++) {
            const uint64_t L = off[r + 1] - off[r];
            ps.get(off[r], L, buf.data());
            CHECK(std::equal(buf.begin(), buf.begin() + L, all.begin() + off[r]), "a read differs");
        }
        // the whole store, then random spans (over block edges)
        std::vector<char> whole(all.size() + 1);
        ps.get(0, all.size(), whole.data());
        CHECK(std::equal(all.begin(), all.end(), whole.begin()), "the whole store differs");
        for (int t = 0; t < 400 && !all.empty(); t++) {
            const uint64_t a = rng() % all.size(), len = rng() % std::min<uint64_t>(all.size() - a, 700) + 1;
            std::vector<char> sp(len);
            ps.get(a, len, sp.data());
            CHECK(std::equal(sp.begin(), sp.end(), all.begin() + a), "a span differs");
        }
        // the 2-bit plane is a quarter of the bases (plus a byte per block at most)
        CHECK(ps.w.size() <= all.size() / 4 + (size_t) nB + 1, "plane larger than 2 bits per base");
    }
    // a read of all N and nothing else
    {
        PackedSeq ps;
        std::vector<char> a(37, 4);
        ps.baseAt = {0, 37}; ps.byteAt = {0, 10}; ps.runs.resize(1); ps.total = 37; ps.w.resize(10);
        PackedSeqWriter w(ps.w.data(), 0, &ps.runs[0]);
        w.put((const uint8_t*) a.data(), 37); w.finish();
        std::vector<char> o(37, 9);
        ps.get(0, 37, o.data());
        CHECK(o == a, "all-N read differs");
        CHECK(ps.runs[0].size() == 1 && ps.runs[0][0].len == 37, "all-N read is one run");
        char one[4]; ps.get(5, 1, one); CHECK(one[0] == 4, "single N inside the run");
    }
    // a code above 4 is reported
    {
        PackedSeq ps; ps.w.resize(2); ps.runs.resize(1);
        PackedSeqWriter w(ps.w.data(), 0, &ps.runs[0]);
        const uint8_t bad[3] = {0, 7, 1};
        w.put(bad, 3);
        CHECK(w.bad, "code 7 not reported");
    }
    if (n_fail == 0) std::printf("test_packed_seq: all passed\n");
    return n_fail == 0 ? 0 : 1;
}
