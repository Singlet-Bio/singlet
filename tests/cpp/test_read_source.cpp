// test_read_source.cpp
// read_source.h: the shared .1fq decoder.
//   - RunSet over two files = one stream (block numbering, read offsets)
//   - format_block = the FASTQ text the FIFO writer wrote (independent reference built from Reader::read_block)
//   - StarChunkSourceT = the bytes STAR's FASTQ branch builds from those FASTQ records, with STAR's chunk boundaries
//   - OrderedBlocks delivers blocks in order whatever the worker timing
//   - binned qualities survive the round trip

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/read_source.h"
#include "singlet/fq/writer.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(expr, name) do { \
    if (expr) { ++g_pass; } \
    else { ++g_fail; std::fprintf(stderr, "  FAIL: %s (line %d)\n", name, __LINE__); } \
} while(0)

using namespace singlet::fq;

struct StarBase {
    virtual ~StarBase() = default;
    virtual void fill(char** chunkIn, uint64_t* sizeBytes, uint64_t sizeLimit, uint64_t& iReadAll,
                      uint64_t readMapNumber, uint32_t readFilesIndex) = 0;
};

static void write_file(const std::string& path, uint32_t n, uint32_t seed, QualMode qm) {
    WriterConfig cfg;
    cfg.block_size = 97;
    cfg.qual_mode = qm;
    cfg.r1_length = 28;
    cfg.r2_length = 0;   // variable
    Writer w;
    w.open(path.c_str(), cfg);
    std::mt19937 rng(seed);
    std::vector<uint8_t> r1(28), r2(120), q(120);
    for (uint32_t i = 0; i < n; ++i) {
        uint16_t l2 = static_cast<uint16_t>(40 + rng() % 60);
        if (i % 50 == 7) l2 = 0;   // empty cDNA read: dropped by the selection
        for (auto& b : r1) b = static_cast<uint8_t>(rng() % 5 == 4 ? 4 : rng() % 4);
        for (uint16_t j = 0; j < l2; ++j) {
            r2[j] = static_cast<uint8_t>(rng() % 4);
            q[j] = static_cast<uint8_t>(rng() % 42);
        }
        w.add_read(r2.data(), l2, q.data(), r1.data(), 28, nullptr);
    }
    w.finish();
}

// Reference: the FIFO writer's FASTQ text, read by the sequential Reader and formatted here independently
static void reference(const std::vector<std::string>& files, int clip, std::string& r1, std::string& r2,
                      std::vector<std::string>& names) {
    static const char A[5] = {'A', 'C', 'G', 'T', 'N'};
    uint64_t roff = 0;
    for (const auto& f : files) {
        Reader rd;
        rd.open(f.c_str());
        DecodedBlock blk;
        while (rd.read_block(blk)) {
            // As the FIFO writer named them: a dropped read (empty mate) does not advance the number; each block starts
            // at its count of stored reads.
            uint64_t num = roff;
            for (uint32_t i = 0; i < blk.n_reads; ++i) {
                const uint16_t l1 = blk.r1_len(i), l2 = blk.r2_len(i);
                if (l1 == 0 || l2 == 0) continue;
                const std::string name = "@r" + std::to_string(num++);
                names.push_back(name);
                r1 += name + "\n";
                for (uint16_t j = 0; j < l1; ++j) r1 += A[blk.r1_seq(i)[j]];
                r1 += "\n+\n" + std::string(l1, 'F') + "\n";
                const int sk = (clip > 0 && clip < l2) ? clip : 0;
                r2 += name + "\n";
                for (uint16_t j = sk; j < l2; ++j) r2 += A[blk.r2_seq(i)[j]];
                r2 += "\n+\n";
                const uint8_t* q = blk.r2_quality(i);
                for (uint16_t j = sk; j < l2; ++j) r2 += q ? static_cast<char>(q[j] + 33) : 'F';
                r2 += "\n";
            }
            roff += blk.n_reads;
        }
    }
}

int main() {
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("test_read_source_" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(dir);
    const std::vector<std::string> files = {dir + "/a.1fq", dir + "/b.1fq"};
    // ILLUMINA4 bins: every Phred lands on one of the four levels, NovaSeq's levels unchanged, packing round-trips
    {
        bool ok = true;
        std::vector<uint8_t> q(46), packed(16), back(46);
        for (int p = 0; p < 46; ++p) q[p] = static_cast<uint8_t>(p);
        pack::pack_qual_4level(q.data(), 46, packed.data(), pack::ilmn4_bin_table());
        pack::unpack_qual_4level(packed.data(), 46, back.data(), qual::kIlmn4Phred);
        for (int p = 0; p < 46; ++p) ok = ok && back[p] == qual::ilmn4_to_phred(qual::phred_to_ilmn4(p));
        for (uint8_t lv : qual::kIlmn4Phred) ok = ok && qual::ilmn4_to_phred(qual::phred_to_ilmn4(lv)) == lv;
        CHECK(ok, "ilmn4_bins");
    }
    write_file(files[0], 1000, 1, QualMode::ILLUMINA4);
    write_file(files[1], 555, 2, QualMode::NONE);   // a run without quality: 'F'

    RunSet runs;
    runs.open(files);
    {
        Reader a, b;
        a.open(files[0].c_str());
        b.open(files[1].c_str());
        CHECK(runs.block_count() == a.block_count() + b.block_count(), "runset_blocks");
        CHECK(runs.stored_reads() == 1555, "runset_reads");
        CHECK(runs.block_read_offset(a.block_count()) == 1000, "runset_offset_second_file");
    }

    for (int clip : {0, 13}) {
        std::string e1, e2;
        std::vector<std::string> names;
        reference(files, clip, e1, e2, names);
        FormatOptions o;
        o.clip5p = clip;
        o.max_r2 = 1024;

        // format_block over every block, in order
        std::string g1, g2;
        for (uint32_t g = 0; g < runs.block_count(); ++g) {
            DecodedBlock blk;
            runs.decode(g, blk);
            FormattedBlock fb;
            format_block(blk, runs.block_read_offset(g), o, fb);
            g1.append(fb.r1.data(), fb.r1.size());
            g2.append(fb.r2.data(), fb.r2.size());
            CHECK(fb.r1_off.size() == fb.n_records() + 1 && fb.r2_off.size() == fb.n_records() + 1, "offsets");
        }
        if (g1 != e1) {
            size_t p = 0;
            while (p < g1.size() && p < e1.size() && g1[p] == e1[p]) ++p;
            std::fprintf(stderr, "  r1 differs at %zu of %zu/%zu:\n  got [%s]\n  ref [%s]\n", p, g1.size(), e1.size(),
                         g1.substr(p > 60 ? p - 60 : 0, 120).c_str(), e1.substr(p > 60 ? p - 60 : 0, 120).c_str());
        }
        CHECK(g1 == e1, "format_r1_equals_reference");
        CHECK(g2 == e2, "format_r2_equals_reference");

        // The STAR chunk source: same records, STAR's name suffix, STAR's chunk boundaries
        for (uint64_t limit : {uint64_t(1), uint64_t(700), uint64_t(1) << 20}) {
            StarChunkSourceT<StarBase> src(files, o, 3, /*cdna_first=*/true);
            std::vector<char> b0(limit + 4096), b1(limit + 4096);
            char* chunk[2] = {b0.data(), b1.data()};
            std::string c0, c1;
            uint64_t iReadAll = 0;
            int fills = 0;
            bool bound_ok = true;
            for (;;) {
                uint64_t sz[2] = {0, 0};
                const uint64_t before = iReadAll;
                src.fill(chunk, sz, limit, iReadAll, uint64_t(-1), 0);
                if (sz[0] == 0) break;
                ++fills;
                // STAR's loop adds reads while both sizes are below the limit: the last read crosses it
                if (!(sz[0] >= limit || sz[1] >= limit) && iReadAll != names.size()) bound_ok = false;
                (void)before;
                c0.append(b0.data(), sz[0]);
                c1.append(b1.data(), sz[1]);
            }
            CHECK(bound_ok, "chunk_boundaries_as_star");
            CHECK(iReadAll == names.size(), "chunk_iReadAll_counts_reads");
            // Expected: STAR's FASTQ branch, "@name iReadAll N 0\n" then seq, '+', qual
            std::string x0, x1;
            size_t p1 = 0, p2 = 0;
            for (size_t k = 0; k < names.size(); ++k) {
                const std::string sfx = " " + std::to_string(k + 1) + " N 0";
                auto take = [](const std::string& s, size_t& p) {
                    size_t e = p;
                    for (int l = 0; l < 4; ++l) e = s.find('\n', e) + 1;
                    std::string r = s.substr(p, e - p);
                    p = e;
                    return r;
                };
                std::string rec1 = take(e1, p1), rec2 = take(e2, p2);
                const size_t nl1 = rec1.find('\n'), nl2 = rec2.find('\n');
                x0 += rec2.substr(0, nl2) + sfx + rec2.substr(nl2);   // mate 0 = cDNA
                x1 += rec1.substr(0, nl1) + sfx + rec1.substr(nl1);   // mate 1 = barcode read
            }
            CHECK(c0 == x0, "chunk_mate0_bytes");
            CHECK(c1 == x1, "chunk_mate1_bytes");
            CHECK(limit != 1 || fills == static_cast<int>(names.size()), "chunk_limit1_one_read_per_fill");
        }

        // readMapNumber stops the source like STAR's own loop
        {
            StarChunkSourceT<StarBase> src(files, o, 2, true);
            std::vector<char> b0(1 << 21), b1(1 << 21);
            char* chunk[2] = {b0.data(), b1.data()};
            uint64_t sz[2] = {0, 0}, iReadAll = 0;
            src.fill(chunk, sz, 1 << 20, iReadAll, 100, 0);
            CHECK(iReadAll == 100, "readMapNumber");
        }
    }

    // OrderedBlocks: in order under random worker delays
    {
        std::vector<uint32_t> got;
        OrderedBlocks<uint32_t> ob(200, 6, 5, [](uint32_t g, uint32_t& v) {
            std::this_thread::sleep_for(std::chrono::microseconds((g * 7919) % 300));
            v = g * 3;
        });
        uint32_t v, b;
        while (ob.next(v, &b)) got.push_back(v + (b * 3 == v ? 0 : 1));
        bool ok = got.size() == 200;
        for (uint32_t i = 0; ok && i < 200; ++i) ok = got[i] == i * 3;
        CHECK(ok, "ordered_blocks_in_order");
    }

    // Pooling refuses runs that are not one layout
    {
        WriterConfig cfg;
        cfg.r1_length = 26;
        Writer w;
        const std::string odd = dir + "/odd.1fq";
        w.open(odd.c_str(), cfg);
        std::vector<uint8_t> r1(26, 1), r2(50, 2);
        w.add_read(r2.data(), 50, nullptr, r1.data(), 26, nullptr);
        w.finish();
        bool threw = false;
        try {
            RunSet r;
            r.open({files[0], odd});
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "runset_rejects_other_layout");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::fprintf(stderr, "test_read_source: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
